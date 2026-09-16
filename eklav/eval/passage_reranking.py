#!/usr/bin/env python3
"""
Passage reranking eval: BRIGHT only (12 domains), vLLM-based generation,
ported from iclr/eval/passage_reranking/eval_cotcond_qwen3_8b.ipynb's
Rank1StyleReranker + BRIGHT runner.

Candidates come from jhu-clsp/rank1-run-files's precomputed BM25 pools
(top-100 per query, gold excluded_ids dropped, no force-injection of missing
gold -- matches the reference notebook exactly).

Scoring: p_true = sigmoid(logit_true - logit_false) from output logprobs,
temperature=0 (greedy), with the incomplete-response fallback (truncate to
last full sentence, force "</think>" continuation, 1-token constrained
decode over {true, false}).

Saves one JSONL per (model, method, domain) with: prompt, raw generation,
parsed score/verdict, gold label, query/doc id, domain, model/checkpoint tag.
Also writes nDCG@10 per domain + the BRIGHT-wide average.
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from eklav.data.eval_downloads import BRIGHT_DOMAINS, bright_runfile_name, load_bright_domain, load_rank1_run_file

SYSTEM_PROMPT = (
    "You are a careful retrieval assistant that judges whether a passage is "
    "relevant to a user's query."
)
TASK_INSTRUCTION = (
    "Determine if the following passage is relevant to the query. "
    "Answer only with 'true' or 'false'."
)

PROMPT_DICT = {
    "BrightRetrieval": (
        "Can you find background information about the concepts used to answer the question:\n\n"
        "FILL_QUERY_HERE\n\n"
        "A passage is relevant if it contains background information about a **sub-concept** that "
        "someone might cite/link to when answering the above question."
    ),
    "BrightRetrieval_aops": (
        "Find different but similar math problems to FILL_QUERY_HERE\n\n"
        "A document is relevant if it uses the same class of functions and shares **any** overlapping techniques."
    ),
    "BrightRetrieval_theoremqa_questions": (
        "Find a passage which uses the same mathematical process as this one: FILL_QUERY_HERE"
    ),
    "BrightRetrieval_leetcode": (
        "I am looking to find different problems that share similar data structures (of any kind) or "
        "algorithms (e.g. DFS, DP, sorting, traversals, etc.). I am looking for problems that share one "
        "or both of these similarities to this:\n\nFILL_QUERY_HERE\n\n"
        "Does this passage share any similarities? e.g. if there was a textbook on leetcode problems, "
        "this would be in the same book even though it could be in a different chapter.\n\n\n"
    ),
    "BrightRetrieval_pony": (
        "I will use the programming language pony. Problem: FILL_QUERY_HERE\n\n"
        "But to solve the problem above, I need to know things about pony. A passage is relevant if it "
        "contains docs that match **any** part (even basic parts) of the code I will have to write for "
        "the above program."
    ),
}
PROMPT_DICT["BrightRetrieval_theoremqa_theorems"] = PROMPT_DICT["BrightRetrieval_theoremqa_questions"]


def get_prompt(domain: str) -> str:
    return PROMPT_DICT.get(f"BrightRetrieval_{domain}", PROMPT_DICT["BrightRetrieval"])


@dataclass
class EvalRecord:
    query_id: str
    doc_id: str
    domain: str
    query: str
    passage: str
    gold_label: int
    prompt: str
    raw_output: str
    parsed_score: float
    parsed_verdict: Optional[str]
    model_tag: str


class Rank1StyleReranker:
    """vLLM-based pointwise reranker: chat-templated prompt with
    enable_thinking=True, temperature=0, p_true = sigmoid(logit_true -
    logit_false) from output logprobs; incomplete-response fallback via a
    1-token constrained decode over {true, false}. Ported from
    eval_cotcond_qwen3_8b.ipynb's Rank1StyleReranker."""

    def __init__(self, model_path: str, num_gpus: int = 1, context_size: int = 20000,
                 max_output_tokens: int = 4096, dtype: str = "bfloat16",
                 gpu_memory_utilization: float = 0.90, lora_path: Optional[str] = None,
                 lora_rank: int = 32):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.model_path = model_path
        self.lora_path = lora_path
        self._lora_request = None

        tokenizer_source = lora_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.true_token_ids = {
            self.tokenizer(" true", add_special_tokens=False).input_ids[0],
            self.tokenizer("true", add_special_tokens=False).input_ids[0],
        }
        self.false_token_ids = {
            self.tokenizer(" false", add_special_tokens=False).input_ids[0],
            self.tokenizer("false", add_special_tokens=False).input_ids[0],
        }

        self.max_output_tokens = max_output_tokens
        self.max_prompt_tokens = context_size - max_output_tokens - 256

        llm_kwargs = dict(
            model=model_path,
            tensor_parallel_size=max(1, int(num_gpus)),
            trust_remote_code=True,
            max_model_len=context_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
        )
        if lora_path:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = lora_rank
        self.model = LLM(**llm_kwargs)

        if lora_path:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest("eval_adapter", 1, lora_path)

        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_output_tokens,
            logprobs=20,
            stop=["</think> true", "</think> false", "</think>\ntrue", "</think>\nfalse",
                  "</think>\n\ntrue", "</think>\n\nfalse"],
            skip_special_tokens=False,
        )

    def _render_chat_prompt(self, query_text: str, passage: str) -> str:
        user_message = f"{TASK_INSTRUCTION}\n\nQuery: {query_text}\nPassage: {passage}"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )

    def format_prompt(self, query: str, passage: str, dataset_prompt: Optional[str] = None) -> str:
        query_text = dataset_prompt.replace("FILL_QUERY_HERE", query) if dataset_prompt else query
        prompt = self._render_chat_prompt(query_text, passage)
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=False).input_ids)
        if prompt_len > self.max_prompt_tokens:
            overflow = prompt_len - self.max_prompt_tokens
            passage_ids = self.tokenizer(passage, add_special_tokens=False).input_ids
            passage = self.tokenizer.decode(passage_ids[: max(0, len(passage_ids) - overflow)])
            prompt = self._render_chat_prompt(query_text, passage)
        return prompt

    def _score_from_output(self, output) -> Optional[float]:
        try:
            final_logits = output.outputs[0].logprobs[-1]
            true_logprobs = [final_logits[t].logprob for t in self.true_token_ids if t in final_logits]
            false_logprobs = [final_logits[t].logprob for t in self.false_token_ids if t in final_logits]
            if not true_logprobs or not false_logprobs:
                return None
            true_score = math.exp(max(true_logprobs))
            false_score = math.exp(max(false_logprobs))
            return true_score / (true_score + false_score)
        except Exception:
            return None

    def _fix_incomplete_responses(self, prompts: List[str], texts: List[str]):
        from vllm import SamplingParams

        cleaned = []
        for text in texts:
            text = text.rstrip()
            if not text.endswith((".", "!", "?")):
                last_punct = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
                if last_punct != -1:
                    text = text[: last_punct + 1]
            cleaned.append(text.strip())

        forced_prompts = [f"{p}\n{c}\n</think>\n\n" for p, c in zip(prompts, cleaned)]
        forced_params = SamplingParams(
            temperature=0, max_tokens=1, logprobs=20,
            allowed_token_ids=list(self.true_token_ids | self.false_token_ids),
            skip_special_tokens=False,
        )
        outputs = self.model.generate(forced_prompts, forced_params, lora_request=self._lora_request)

        fixed_texts, fixed_scores = [], []
        for cleaned_text, output in zip(cleaned, outputs):
            score = self._score_from_output(output)
            forced_text = output.outputs[0].text if output.outputs else ""
            fixed_texts.append(f"\n{cleaned_text}\n</think>{forced_text}")
            fixed_scores.append(score if score is not None else 0.5)
        return fixed_texts, fixed_scores

    def generate_and_score(self, prompts: List[str]):
        if not prompts:
            return []
        outputs = self.model.generate(prompts, self.sampling_params, lora_request=self._lora_request)

        texts: List[str] = [""] * len(prompts)
        scores: List[Optional[float]] = [None] * len(prompts)
        incomplete_idx = []
        for i, output in enumerate(outputs):
            texts[i] = output.outputs[0].text if output.outputs else ""
            score = self._score_from_output(output)
            if score is None:
                incomplete_idx.append(i)
            else:
                scores[i] = score

        if incomplete_idx:
            fixed_texts, fixed_scores = self._fix_incomplete_responses(
                [prompts[i] for i in incomplete_idx], [texts[i] for i in incomplete_idx]
            )
            for j, i in enumerate(incomplete_idx):
                texts[i] = texts[i] + fixed_texts[j]
                scores[i] = fixed_scores[j]

        return list(zip(texts, scores))


def run_bright_domain(model: Rank1StyleReranker, domain: str, out_dir: str, model_tag: str,
                       top_k: int = 100, num_questions: Optional[int] = None) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    corpus, queries, qrels, excluded = load_bright_domain(domain)

    sampled_qids = sorted(queries)[:num_questions] if num_questions else sorted(queries)
    queries = {qid: queries[qid] for qid in sampled_qids}
    qrels = {qid: qrels[qid] for qid in sampled_qids}
    excluded = {qid: excluded.get(qid, set()) for qid in sampled_qids}

    run = load_rank1_run_file(bright_runfile_name(domain))
    top_ranked: Dict[str, List[str]] = {}
    for qid, cands in run.items():
        if qid not in queries:
            continue
        excl = excluded.get(qid, set())
        ranked = sorted(cands.items(), key=lambda kv: kv[1], reverse=True)
        top_ranked[qid] = [cid for cid, _ in ranked if cid not in excl][:top_k]

    dataset_prompt = get_prompt(domain)
    pairs, pair_ids = [], []
    for qid, cids in top_ranked.items():
        for cid in cids:
            if cid not in corpus:
                continue
            pairs.append((queries[qid], corpus[cid]))
            pair_ids.append((qid, cid))

    prompts = [model.format_prompt(q, p, dataset_prompt=dataset_prompt) for q, p in pairs]
    results = model.generate_and_score(prompts)

    from eklav.eval.metrics import ndcg_at_k

    scored: Dict[str, Dict[str, float]] = {qid: {} for qid in queries}
    jsonl_path = os.path.join(out_dir, f"{domain}.jsonl")
    with open(jsonl_path, "w") as f:
        for (qid, cid), prompt, (query, passage), (raw_text, score) in zip(pair_ids, prompts, pairs, results):
            score = score if score is not None else 0.5
            scored[qid][cid] = score
            verdict = "true" if score >= 0.5 else "false"
            gold_label = qrels.get(qid, {}).get(cid, 0)
            record = {
                "query_id": qid, "doc_id": cid, "domain": domain,
                "query": query, "passage": passage,
                "prompt": prompt, "raw_output": raw_text,
                "parsed_score": score, "parsed_verdict": verdict,
                "gold_label": gold_label,
                "model_tag": model_tag, "benchmark": "BRIGHT",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    ndcg_per_query = []
    for qid, cand_scores in scored.items():
        gold = {d for d, s in qrels.get(qid, {}).items() if s > 0}
        ranked = [d for d, _ in sorted(cand_scores.items(), key=lambda kv: kv[1], reverse=True)]
        ndcg_per_query.append(ndcg_at_k(ranked, gold, 10))
    ndcg10 = sum(ndcg_per_query) / len(ndcg_per_query) if ndcg_per_query else 0.0

    summary = {"domain": domain, "ndcg_at_10": ndcg10, "n_queries": len(queries)}
    with open(os.path.join(out_dir, f"{domain}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Merged model dir, or base model if --lora_path given.")
    ap.add_argument("--lora_path", default=None)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--model_tag", required=True, help="Identifier for output files, e.g. qwen3_8b_eklav_ckpt5968")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_gpus", type=int, default=1)
    ap.add_argument("--context_size", type=int, default=20000)
    ap.add_argument("--max_output_tokens", type=int, default=4096)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--num_questions", type=int, default=None)
    ap.add_argument("--domains", nargs="*", default=None, help="Subset of BRIGHT domains (default: all 12)")
    args = ap.parse_args()

    domains = args.domains or BRIGHT_DOMAINS
    model = Rank1StyleReranker(
        model_path=args.model_path, num_gpus=args.num_gpus, context_size=args.context_size,
        max_output_tokens=args.max_output_tokens, lora_path=args.lora_path, lora_rank=args.lora_rank,
    )

    results = {}
    for domain in domains:
        t0 = time.time()
        results[domain] = run_bright_domain(
            model, domain, args.out_dir, args.model_tag, top_k=args.top_k, num_questions=args.num_questions
        )
        print(f"[{domain}] nDCG@10={results[domain]['ndcg_at_10']:.4f} ({time.time() - t0:.0f}s)")

    if len(results) == len(BRIGHT_DOMAINS):
        avg = sum(r["ndcg_at_10"] for r in results.values()) / len(results)
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump({"per_domain": results, "avg_ndcg_at_10": avg}, f, indent=2)
        print(f"BRIGHT average nDCG@10 = {avg:.4f}")


if __name__ == "__main__":
    main()
