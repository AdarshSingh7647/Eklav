"""
Table reranking evaluation (listwise). For each query a candidate pool is ranked
with a sliding window:

  1. Pool: the benchmark's candidates if it ships them, else the BM25 + SPLADE + mpnet
     reciprocal-rank-fusion pool (top-100) with any missing relevant table appended.
  2. Windows of <= 20 tables / <= 20,000 prompt tokens, stride 10.
  3. The model reasons, then emits {"ranked_tables": [...]} per window; each window's
     returned order replaces that slice of the master list (bubble-sort aggregation).
  4. A query with any unparseable window scores zero; the parse-failure count is reported.

No teacher context is ever supplied at evaluation.
"""

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from eklav.eval.benchmarks import RetrievalData
from eklav.eval.metrics import compute_ranking_metrics, mean_metrics, zero_metrics
from eklav.eval.model import ModelArgs, load_llm, load_tokenizer
from eklav.tasks.table_reranking import SYSTEM_PROMPT

EVAL_K = (1, 5, 10, 20)
MAX_DOC_CHARS = 1500
POOL_SIZE = 100
WINDOW_SIZE = 20
WINDOW_TOKENS = 20000
STRIDE = 10
DEFAULT_MAX_MODEL_LEN = 32768
INSTRUCTION = (
    "\n\nReason through each table's relevance step-by-step.\n"
    'Then output exactly: {"ranked_tables": [1, 2, 3, ...]}'
)


def _chat(messages: List[dict], tokenizer) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def window_user_message(question: str, window_ids: List[str], corpus: Dict[str, str]) -> str:
    parts = [f"Question: {question.strip()}\n\nCandidate tables to rank:\n"]
    for i, doc_id in enumerate(window_ids, 1):
        parts.append(f"\n### Table {i}\n{corpus[doc_id][:MAX_DOC_CHARS]}")
    parts.append(INSTRUCTION)
    return "".join(parts)


def window_overhead(question: str, tokenizer) -> int:
    user = f"Question: {question.strip()}\n\nCandidate tables to rank:\n{INSTRUCTION}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    return count_tokens(_chat(messages, tokenizer), tokenizer)


def build_windows(pool: List[str], question: str, corpus: Dict[str, str], table_tokens: Dict[str, int],
                  tokenizer, window_size: int = WINDOW_SIZE, window_tokens: int = WINDOW_TOKENS,
                  stride: int = STRIDE) -> List[Dict]:
    overhead = window_overhead(question, tokenizer)
    windows, start, idx, n = [], 0, 0, len(pool)
    while start < n:
        window_ids: List[str] = []
        total = overhead
        for pos in range(start, n):
            doc_id = pool[pos]
            tokens = table_tokens.get(doc_id, 0)
            if len(window_ids) >= window_size:
                break
            if window_ids and total + tokens > window_tokens:
                break
            window_ids.append(doc_id)
            total += tokens
        if not window_ids:
            window_ids = [pool[start]]
        user = window_user_message(question, window_ids, corpus)
        prompt = _chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}], tokenizer)
        windows.append({"window_idx": idx, "window_start": start, "window_end": start + len(window_ids),
                        "window_candidate_ids": window_ids, "prompt": prompt})
        start += stride
        idx += 1
    return windows


def parse_thinking_and_answer(text: str) -> Tuple[str, str]:
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return "", text.strip()


def parse_ranked_tables(text: str) -> Optional[List[int]]:
    """The last {"ranked_tables": [...]} object in `text` (falling back to a bare
    "ranked_tables": [...] fragment) as a list of 1-based table numbers."""
    found = None
    for m in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and isinstance(obj.get("ranked_tables"), list) and obj["ranked_tables"]:
                found = obj["ranked_tables"]
        except json.JSONDecodeError:
            pass
    if found is None:
        m = re.search(r'"ranked_tables"\s*:\s*\[([^\]]*)\]', text, flags=re.DOTALL)
        if m:
            found = [p.strip().strip("\"'") for p in m.group(1).split(",") if p.strip()]
    if not found:
        return None
    ranks = []
    for x in found:
        try:
            ranks.append(int(x))
        except (ValueError, TypeError):
            pass
    return ranks or None


def aggregate_bubble_sort(initial_pool: List[str], windows: List[Dict]) -> Tuple[List[str], bool]:
    """Final ranking after each window's order replaces its slice of the pool, and
    whether any window failed to parse (failed windows leave their slice unchanged)."""
    master, any_failed = list(initial_pool), False
    for w in sorted(windows, key=lambda w: w["window_idx"]):
        candidates, order = w["window_candidate_ids"], w.get("ranked_indices")
        if w.get("parse_failed", order is None) or not order:
            any_failed = True
            continue
        seen: Set[str] = set()
        ordered: List[str] = []
        for ix in order:
            if 1 <= ix <= len(candidates) and candidates[ix - 1] not in seen:
                ordered.append(candidates[ix - 1])
                seen.add(candidates[ix - 1])
        ordered += [d for d in candidates if d not in seen]
        master[w["window_start"]:w["window_end"]] = ordered
    return master, any_failed


@dataclass
class Prepared:
    benchmark: str
    subset: str
    data: RetrievalData
    qids: List[str]
    pools: Dict[str, List[str]]
    records: List[Dict]              # one per window prompt


class TableRerankingEvaluator:
    """Evaluate a model on any retrieval benchmark, listwise."""

    # Hybrid first-stage models share the GPU with vLLM, so every benchmark is
    # prepared before the engine is loaded.
    prepare_all_first = True

    def __init__(self, args: ModelArgs, first_stage: str = "hybrid", skip_splade: bool = False,
                 retrieval_device: str = "cuda:0", **_ignored):
        self.args = args
        self.first_stage = first_stage
        self.skip_splade = skip_splade
        self.retrieval_device = retrieval_device
        self.tokenizer = load_tokenizer(args)
        self.context = args.max_model_len or DEFAULT_MAX_MODEL_LEN
        self.llm = None
        self.lora_request = None

    def prepare(self, benchmark: str, subset: str, data: RetrievalData, qids: List[str]) -> Prepared:
        if data.candidates is not None:
            pools = {q: data.candidates.get(q, [])[:POOL_SIZE] for q in qids}
        else:
            from eklav.eval.retrieval import build_pools

            pools = build_pools(data.corpus, {q: data.queries[q] for q in qids},
                                {q: list(data.qrels[q]) for q in qids}, kind=self.first_stage, top_k=POOL_SIZE,
                                inject_gold=True, device=self.retrieval_device, skip_splade=self.skip_splade)
        table_tokens = {d: count_tokens(f"\n### Table 1\n{data.corpus[d][:MAX_DOC_CHARS]}", self.tokenizer)
                        for pool in pools.values() for d in pool if d in data.corpus}
        records = []
        for qid in qids:
            pool = [d for d in pools[qid] if d in data.corpus]
            pools[qid] = pool
            for w in build_windows(pool, data.queries[qid], data.corpus, table_tokens, self.tokenizer):
                records.append({"qid": qid, **w})
        return Prepared(benchmark, subset, data, qids, pools, records)

    def _generate(self, prompts: List[str]):
        if self.llm is None:
            from vllm import SamplingParams

            self.llm, self.lora_request = load_llm(self.args, self.context)
            self.sampling = SamplingParams(temperature=0.0, max_tokens=self.args.max_output_tokens)
        return self.llm.generate(prompts, self.sampling, lora_request=self.lora_request)

    def run(self, prepared: Prepared, out_dir: str, model_tag: str) -> Dict:
        data = prepared.data
        outputs = self._generate([r["prompt"] for r in prepared.records])

        windows: Dict[str, List[Dict]] = defaultdict(list)
        for rec, output in zip(prepared.records, outputs):
            raw = output.outputs[0].text.strip()
            thinking, answer = parse_thinking_and_answer(raw)
            order = parse_ranked_tables(answer) or parse_ranked_tables(raw)
            windows[rec["qid"]].append({
                "window_idx": rec["window_idx"], "window_start": rec["window_start"], "window_end": rec["window_end"],
                "window_candidate_ids": rec["window_candidate_ids"], "ranked_indices": order,
                "parse_failed": order is None, "thinking": thinking, "answer_text": answer,
                "prompt": rec["prompt"], "raw_output": raw,
            })

        rows, per_query, n_failed = [], [], 0
        for qid in prepared.qids:
            ws = sorted(windows[qid], key=lambda w: w["window_idx"])
            gold = set(data.qrels[qid])
            ranking, failed = aggregate_bubble_sort(prepared.pools[qid], ws)
            n_failed += failed
            metrics = zero_metrics(EVAL_K) if failed else compute_ranking_metrics(ranking, gold, EVAL_K)
            per_query.append(metrics)
            positions = {d: i + 1 for i, d in enumerate(ranking)}
            rows.append({
                "qid": qid, "benchmark": prepared.benchmark, "subset": prepared.subset,
                "question": data.queries[qid], "gold_doc_ids": sorted(gold), "final_ranked_list": ranking,
                "rank_of_gold_after_reranking": {g: positions.get(g) for g in gold},
                "any_parse_failed": failed, "n_windows": len(ws),
                "windows": [{k: w[k] for k in ("window_idx", "prompt", "raw_output", "thinking", "answer_text",
                                               "ranked_indices", "parse_failed")} for w in ws],
                "metrics": metrics, "model_tag": model_tag,
            })

        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{prepared.subset}.jsonl"), "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"benchmark": prepared.benchmark, "subset": prepared.subset, "n_queries": len(rows),
                **mean_metrics(per_query), "n_parse_failed_queries": n_failed}
