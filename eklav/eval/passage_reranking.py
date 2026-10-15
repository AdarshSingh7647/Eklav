"""
Passage reranking evaluation (pointwise). The model judges each (query, passage)
pair; reranking order is p(true) = sigmoid(logit_true - logit_false) read from the
logprobs of the verdict token, with temperature 0. A response that does not reach a
verdict (e.g. cut off mid-reasoning) is completed by truncating to its last full
sentence, forcing "</think>", and decoding one constrained token over {true, false}.

Candidates are the benchmark's precomputed pools when it ships them (BRIGHT: the
BM25 top-100 pools released with Rank1) and a BM25 top-k pool otherwise. No teacher
context is ever supplied at evaluation.
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from eklav.eval.benchmarks import RetrievalData
from eklav.eval.metrics import compute_ranking_metrics, mean_metrics
from eklav.eval.model import ModelArgs, load_llm, load_tokenizer
from eklav.tasks.passage_reranking import SYSTEM_PROMPT

TASK_INSTRUCTION = (
    "Determine if the following passage is relevant to the query. "
    "Answer only with 'true' or 'false'."
)
EVAL_K = (1, 5, 10, 20)
DEFAULT_MAX_MODEL_LEN = 20000
STOP = ["</think> true", "</think> false", "</think>\ntrue", "</think>\nfalse",
        "</think>\n\ntrue", "</think>\n\nfalse"]


class PointwiseScorer:
    """Chat-templated pointwise scorer over a vLLM engine (loaded lazily by `load`)."""

    def __init__(self, args: ModelArgs):
        self.args = args
        self.tokenizer = load_tokenizer(args)
        self.true_ids = {self.tokenizer(t, add_special_tokens=False).input_ids[0] for t in (" true", "true")}
        self.false_ids = {self.tokenizer(t, add_special_tokens=False).input_ids[0] for t in (" false", "false")}
        self.context = args.max_model_len or DEFAULT_MAX_MODEL_LEN
        self.max_prompt_tokens = self.context - args.max_output_tokens - 256
        self.llm = None
        self.lora_request = None

    def load(self) -> None:
        if self.llm is not None:
            return
        from vllm import SamplingParams

        self.llm, self.lora_request = load_llm(self.args, self.context)
        self.sampling = SamplingParams(temperature=0, max_tokens=self.args.max_output_tokens, logprobs=20,
                                       stop=STOP, skip_special_tokens=False)

    def _render(self, query: str, passage: str) -> str:
        user = f"{TASK_INSTRUCTION}\n\nQuery: {query}\nPassage: {passage}"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=True)

    def format_prompt(self, query: str, passage: str) -> str:
        """The chat prompt, with the passage truncated if the prompt would not fit."""
        prompt = self._render(query, passage)
        length = len(self.tokenizer(prompt, add_special_tokens=False).input_ids)
        if length > self.max_prompt_tokens:
            ids = self.tokenizer(passage, add_special_tokens=False).input_ids
            passage = self.tokenizer.decode(ids[: max(0, len(ids) - (length - self.max_prompt_tokens))])
            prompt = self._render(query, passage)
        return prompt

    def _score(self, output) -> Optional[float]:
        try:
            last = output.outputs[0].logprobs[-1]
            true_lp = [last[t].logprob for t in self.true_ids if t in last]
            false_lp = [last[t].logprob for t in self.false_ids if t in last]
            if not true_lp or not false_lp:
                return None
            t, f = math.exp(max(true_lp)), math.exp(max(false_lp))
            return t / (t + f)
        except Exception:
            return None

    def _complete(self, prompts: List[str], texts: List[str]) -> Tuple[List[str], List[float]]:
        from vllm import SamplingParams

        cleaned = []
        for text in texts:
            text = text.rstrip()
            if not text.endswith((".", "!", "?")):
                last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
                if last != -1:
                    text = text[: last + 1]
            cleaned.append(text.strip())
        forced = SamplingParams(temperature=0, max_tokens=1, logprobs=20,
                                allowed_token_ids=list(self.true_ids | self.false_ids), skip_special_tokens=False)
        outputs = self.llm.generate([f"{p}\n{c}\n</think>\n\n" for p, c in zip(prompts, cleaned)], forced,
                                    lora_request=self.lora_request)
        out_texts, scores = [], []
        for text, output in zip(cleaned, outputs):
            score = self._score(output)
            out_texts.append(f"\n{text}\n</think>{output.outputs[0].text if output.outputs else ''}")
            scores.append(score if score is not None else 0.5)
        return out_texts, scores

    def generate_and_score(self, prompts: List[str]) -> List[Tuple[str, float]]:
        if not prompts:
            return []
        self.load()
        outputs = self.llm.generate(prompts, self.sampling, lora_request=self.lora_request)
        texts = [o.outputs[0].text if o.outputs else "" for o in outputs]
        scores: List[Optional[float]] = [self._score(o) for o in outputs]
        incomplete = [i for i, s in enumerate(scores) if s is None]
        if incomplete:
            fixed_texts, fixed_scores = self._complete([prompts[i] for i in incomplete], [texts[i] for i in incomplete])
            for j, i in enumerate(incomplete):
                texts[i] += fixed_texts[j]
                scores[i] = fixed_scores[j]
        return list(zip(texts, scores))


@dataclass
class Prepared:
    benchmark: str
    subset: str
    data: RetrievalData
    qids: List[str]
    pairs: List[Tuple[str, str]]     # (query id, doc id)
    prompts: List[str]


class PassageRerankingEvaluator:
    """Evaluate a model on any retrieval benchmark, pointwise."""

    prepare_all_first = False        # preparation needs no GPU, so it can interleave with generation

    def __init__(self, args: ModelArgs, top_k: int = 100, first_stage: str = "bm25", **_ignored):
        self.scorer = PointwiseScorer(args)
        self.top_k = top_k
        self.first_stage = first_stage

    def prepare(self, benchmark: str, subset: str, data: RetrievalData, qids: List[str]) -> Prepared:
        if data.candidates is not None:
            pools = {q: data.candidates.get(q, []) for q in qids}
        else:
            from eklav.eval.retrieval import build_pools

            pools = build_pools(data.corpus, {q: data.queries[q] for q in qids}, {}, kind=self.first_stage,
                                top_k=self.top_k)
        template = data.query_template
        pairs, prompts = [], []
        for qid in qids:
            query = data.queries[qid]
            query_text = template.replace("{query}", query) if template else query
            for doc_id in pools[qid][: self.top_k]:
                if doc_id in data.corpus:
                    pairs.append((qid, doc_id))
                    prompts.append(self.scorer.format_prompt(query_text, data.corpus[doc_id]))
        return Prepared(benchmark, subset, data, qids, pairs, prompts)

    def run(self, prepared: Prepared, out_dir: str, model_tag: str) -> Dict:
        data = prepared.data
        results = self.scorer.generate_and_score(prepared.prompts)

        scored: Dict[str, Dict[str, float]] = {q: {} for q in prepared.qids}
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{prepared.subset}.jsonl"), "w") as f:
            for (qid, doc_id), prompt, (raw, score) in zip(prepared.pairs, prepared.prompts, results):
                score = 0.5 if score is None else score
                scored[qid][doc_id] = score
                f.write(json.dumps({
                    "query_id": qid, "doc_id": doc_id, "benchmark": prepared.benchmark, "subset": prepared.subset,
                    "query": data.queries[qid], "passage": data.corpus[doc_id], "prompt": prompt,
                    "raw_output": raw, "parsed_score": score, "parsed_verdict": "true" if score >= 0.5 else "false",
                    "gold_label": data.qrels.get(qid, {}).get(doc_id, 0), "model_tag": model_tag,
                }, ensure_ascii=False) + "\n")

        per_query = []
        for qid, doc_scores in scored.items():
            ranked = [d for d, _ in sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)]
            per_query.append(compute_ranking_metrics(ranked, set(data.qrels.get(qid, {})), EVAL_K))
        return {"benchmark": prepared.benchmark, "subset": prepared.subset, "n_queries": len(prepared.qids),
                **mean_metrics(per_query)}
