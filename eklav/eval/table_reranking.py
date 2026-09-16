#!/usr/bin/env python3
"""
Table reranking eval: full RRF + sliding-window pipeline, ported faithfully
from model-forge/code/tabular_reranking/eval/ood_eval/build_prompts_v3.py and
run_full_eval_v3.py (author-confirmed full-fidelity port, not simplified).

Pipeline:
  1. BM25 + SPLADE + mpnet retrieval, top-100 per query, per retriever.
  2. RRF fusion (k=60) into a single top-100 pool; inject any missing gold doc.
  3. Sliding window over the pool: <=20 tables/window, stride 10, <=20000
     prompt tokens/window.
  4. Bubble-sort-style aggregation: each window's model-returned order
     replaces that slice of the master list; final rank = position in the
     resulting master list after all windows are processed.
  5. nDCG@10 (+ recall/precision/MRR/perfect-recall @ {1,5,10,20}) via the
     same binary-relevance formula as the reference.

Works against any of AITQARetrieval, FeTaQARetrieval, MultiHierttRetrieval,
OTTQASmallRetrieval, OpenWikiTablesRetrieval (OOD) or NQTablesRetrieval
(in-domain) via --benchmark.

Saves one JSONL per (model, method, benchmark): one row per query, with
prompt/thinking/answer per window, final ranking, gold ids, and metrics.
"""

import argparse
import gc
import json
import math
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from eklav.data.table_reranking import OOD_BENCHMARK_REPOS
from eklav.eval.metrics import compute_ranking_metrics

MAX_DOC_CHARS = 1500
EVAL_K = [1, 5, 10, 20]

TOP_K = 100
RRF_TOP_K = 100
RRF_K = 60
WINDOW_SIZE = 20
WINDOW_TOKENS = 20000
STRIDE = 10

SYSTEM_PROMPT = (
    "You are a careful retrieval assistant that ranks candidate tables by "
    "their relevance to a user's question."
)
INSTRUCTION = (
    '\n\nReason through each table\'s relevance step-by-step.\n'
    'Then output exactly: {"ranked_tables": [1, 2, 3, ...]}'
)

MPNET_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
SPLADE_MODEL_ID = "naver/splade-v3"


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #

def load_ibm_dataset(benchmark: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, int]]]:
    from datasets import load_dataset

    if benchmark not in OOD_BENCHMARK_REPOS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    hf_id = OOD_BENCHMARK_REPOS[benchmark]

    qrels_ds = None
    for split in ("test", "dev", "train"):
        try:
            qrels_ds = load_dataset(hf_id, "default")[split]
            break
        except Exception:
            continue
    if qrels_ds is None:
        raise RuntimeError(f"Could not load qrels for {benchmark}")
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for row in qrels_ds:
        qrels[str(row["qid"])][str(row["did"])] = int(row["score"])

    queries_ds = None
    for split in ("test_queries", "dev_queries", "train_queries"):
        try:
            queries_ds = load_dataset(hf_id, "queries")[split]
            break
        except Exception:
            continue
    if queries_ds is None:
        raise RuntimeError(f"Could not load queries for {benchmark}")
    queries = {str(r["_id"]): r["text"] for r in queries_ds}

    corpus_ds = None
    for cfg in ("corpus_md", "corpus_linearized", "corpus"):
        try:
            corpus_ds = load_dataset(hf_id, cfg)[cfg]
            break
        except Exception:
            continue
    if corpus_ds is None:
        raise RuntimeError(f"Could not load corpus for {benchmark}")

    def _doc_text(row) -> str:
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        return f"{title}\n{text}" if title and text else (title or text)

    corpus = {str(r["_id"]): _doc_text(r) for r in corpus_ds}
    return queries, corpus, dict(qrels)


def sample_queries(qrels: Dict, queries: Dict, n: Optional[int], seed: int) -> List[str]:
    import random

    valid = [qid for qid in qrels if qid in queries and len(qrels[qid]) > 0]
    if n is None or n >= len(valid):
        return sorted(valid)
    return sorted(random.Random(seed).sample(valid, n))


# --------------------------------------------------------------------------- #
# Retrieval: BM25 + SPLADE + mpnet, RRF fusion
# --------------------------------------------------------------------------- #

def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def build_bm25(corpus_ids: List[str], corpus: Dict[str, str]):
    from rank_bm25 import BM25Okapi

    return BM25Okapi([_tokenize(corpus[d]) for d in corpus_ids])


def bm25_search(bm25, corpus_ids: List[str], query: str, top_k: int) -> List[Tuple[str, float]]:
    import numpy as np

    scores = bm25.get_scores(_tokenize(query))
    n = min(top_k, len(scores))
    idx = np.argpartition(scores, -n)[-n:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [(corpus_ids[i], float(scores[i])) for i in idx]


class SPLADEEncoder:
    def __init__(self, model_id: str, device: str):
        from sentence_transformers import SparseEncoder

        self.model = SparseEncoder(model_id, device=device)

    def encode_corpus(self, texts: List[str], batch_size: int = 128):
        import scipy.sparse as sp

        emb = self.model.encode_document(texts, batch_size=batch_size, show_progress_bar=True)
        t = emb.to_dense() if emb.is_sparse else emb
        return sp.csr_matrix(t.cpu().float().numpy())

    def encode_query(self, text: str):
        import scipy.sparse as sp

        emb = self.model.encode_query([text], show_progress_bar=False)
        t = emb.to_dense() if emb.is_sparse else emb
        return sp.csr_matrix(t.cpu().float().numpy())


def splade_search(query_vec, doc_matrix, corpus_ids: List[str], top_k: int) -> List[Tuple[str, float]]:
    import numpy as np

    scores = (doc_matrix @ query_vec.T).toarray().flatten()
    n = min(top_k, len(scores))
    idx = np.argpartition(scores, -n)[-n:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [(corpus_ids[i], float(scores[i])) for i in idx]


class MPNetEncoder:
    def __init__(self, model_id: str, device: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id, device=device)

    def encode(self, texts: List[str], batch_size: int = 256):
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                                  normalize_embeddings=True, convert_to_numpy=True)


def mpnet_search(query_vec, doc_matrix, corpus_ids: List[str], top_k: int) -> List[Tuple[str, float]]:
    import numpy as np

    scores = (doc_matrix @ query_vec).flatten()
    n = min(top_k, len(scores))
    idx = np.argpartition(scores, -n)[-n:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [(corpus_ids[i], float(scores[i])) for i in idx]


def rrf_merge(ranked_lists: List[List[Tuple[str, float]]], top_k: int, k: int = 60) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (did, _) in enumerate(ranked, 1):
            scores[did] += 1.0 / (k + rank)
    return [did for did, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]


def build_candidate_pool(bm25_ranked, splade_ranked, mpnet_ranked, gold_ids: List[str],
                          rrf_top_k: int, rrf_k: int) -> Tuple[List[str], int]:
    ranked_lists = [l for l in [bm25_ranked, splade_ranked, mpnet_ranked] if l]
    pool = rrf_merge(ranked_lists, top_k=rrf_top_k, k=rrf_k)
    pool_set = set(pool)
    missing = [g for g in gold_ids if g not in pool_set]
    if missing:
        pool = pool + missing
    return pool, len(missing)


# --------------------------------------------------------------------------- #
# Sliding-window prompt construction
# --------------------------------------------------------------------------- #

def _apply_chat_template(messages: List[dict], tokenizer, enable_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=enable_thinking)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_user_message_for_window(question: str, window_ids: List[str], corpus: Dict[str, str]) -> str:
    parts = [f"Question: {question.strip()}\n\nCandidate tables to rank:\n"]
    for i, did in enumerate(window_ids, 1):
        parts.append(f"\n### Table {i}\n{corpus[did][:MAX_DOC_CHARS]}")
    parts.append(INSTRUCTION)
    return "".join(parts)


def compute_window_overhead(question: str, tokenizer) -> int:
    user_msg = f"Question: {question.strip()}\n\nCandidate tables to rank:\n{INSTRUCTION}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
    return count_tokens(_apply_chat_template(messages, tokenizer, enable_thinking=True), tokenizer)


def build_windows(pool: List[str], question: str, corpus: Dict[str, str], table_tokens: Dict[str, int],
                   tokenizer, window_size: int = WINDOW_SIZE, window_tokens: int = WINDOW_TOKENS,
                   stride: int = STRIDE) -> List[Dict]:
    overhead = compute_window_overhead(question, tokenizer)
    windows, start, idx, n = [], 0, 0, len(pool)

    while start < n:
        window_ids: List[str] = []
        token_sum = overhead
        for pos in range(start, n):
            did = pool[pos]
            tok = table_tokens.get(did, 0)
            if len(window_ids) >= window_size:
                break
            if window_ids and token_sum + tok > window_tokens:
                break
            window_ids.append(did)
            token_sum += tok
        if not window_ids:
            window_ids = [pool[start]]

        user_message = build_user_message_for_window(question, window_ids, corpus)
        prompt = _apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
            tokenizer, enable_thinking=True,
        )
        windows.append({
            "window_idx": idx, "window_start": start, "window_end": start + len(window_ids),
            "window_candidate_ids": window_ids, "user_message": user_message, "prompt": prompt,
            "total_window_input_tokens": count_tokens(prompt, tokenizer),
        })
        start += stride
        idx += 1
        if start >= n:
            break
    return windows


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #

def parse_thinking_and_answer(text: str) -> Tuple[str, str]:
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return "", text.strip()


def parse_ranked_tables(text: str) -> Optional[List[int]]:
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
    out = []
    for x in found:
        try:
            out.append(int(x))
        except (ValueError, TypeError):
            pass
    return out if out else None


# --------------------------------------------------------------------------- #
# Bubble-sort aggregation
# --------------------------------------------------------------------------- #

def aggregate_bubble_sort(initial_pool: List[str], window_outputs: List[Dict]) -> Tuple[List[str], bool]:
    master = list(initial_pool)
    any_failed = False
    for w in sorted(window_outputs, key=lambda x: x["window_idx"]):
        cands = w["window_candidate_ids"]
        ranked_ix = w.get("ranked_indices")
        parse_fail = w.get("parse_failed", ranked_ix is None)
        start, end = w["window_start"], w["window_end"]
        if parse_fail or not ranked_ix:
            any_failed = True
            continue
        n = len(cands)
        seen: Set[str] = set()
        ordered: List[str] = []
        for ix in ranked_ix:
            if 1 <= ix <= n:
                did = cands[ix - 1]
                if did not in seen:
                    ordered.append(did)
                    seen.add(did)
        for did in cands:
            if did not in seen:
                ordered.append(did)
        master[start:end] = ordered
    return master, any_failed


def gold_ranks(ordered_ids: List[str], gold_ids: Set[str]) -> Dict[str, Optional[int]]:
    rank_map = {did: i + 1 for i, did in enumerate(ordered_ids)}
    return {gid: rank_map.get(gid) for gid in gold_ids}


# --------------------------------------------------------------------------- #
# Main eval driver
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=list(OOD_BENCHMARK_REPOS))
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lora_path", default=None)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_questions", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.88)
    ap.add_argument("--skip_splade", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"[{args.benchmark}] loading dataset ...")
    queries, corpus, qrels = load_ibm_dataset(args.benchmark)
    qids = sample_queries(qrels, queries, args.num_questions, args.seed)
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[d] for d in corpus_ids]

    table_tokens = {did: count_tokens(f"\n### Table 1\n{corpus[did][:MAX_DOC_CHARS]}", tokenizer) for did in corpus_ids}

    print(f"[{args.benchmark}] building BM25 index on {len(corpus_ids)} docs ...")
    bm25 = build_bm25(corpus_ids, corpus)

    splade_enc = None
    splade_matrix = None
    if not args.skip_splade:
        try:
            splade_enc = SPLADEEncoder(SPLADE_MODEL_ID, "cuda:0")
            splade_matrix = splade_enc.encode_corpus(corpus_texts)
        except Exception as e:
            print(f"[{args.benchmark}] SPLADE unavailable ({e}), continuing with BM25+mpnet only")

    mpnet_enc = MPNetEncoder(MPNET_MODEL_ID, "cuda:0")
    mpnet_matrix = mpnet_enc.encode(corpus_texts)

    all_records = []
    query_meta: Dict[str, Dict] = {}
    for qid in qids:
        question = queries[qid]
        gold_ids = list(qrels[qid].keys())

        bm25_ranked = bm25_search(bm25, corpus_ids, question, TOP_K)
        splade_ranked = []
        if splade_matrix is not None:
            qvec = splade_enc.encode_query(question)
            splade_ranked = splade_search(qvec, splade_matrix, corpus_ids, TOP_K)
        qvec_mpnet = mpnet_enc.model.encode(question, normalize_embeddings=True, convert_to_numpy=True,
                                             show_progress_bar=False)
        mpnet_ranked = mpnet_search(qvec_mpnet, mpnet_matrix, corpus_ids, TOP_K)

        pool, _n_injected = build_candidate_pool(bm25_ranked, splade_ranked, mpnet_ranked, gold_ids, RRF_TOP_K, RRF_K)
        windows = build_windows(pool, question, corpus, table_tokens, tokenizer,
                                 window_size=WINDOW_SIZE, window_tokens=WINDOW_TOKENS, stride=STRIDE)

        query_meta[qid] = {"question": question, "gold_ids": gold_ids, "pool": pool}
        for w in windows:
            all_records.append({"qid": qid, **w})

    del splade_matrix, mpnet_matrix, bm25
    gc.collect()

    print(f"[{args.benchmark}] loading vLLM model ...")
    llm = LLM(model=args.model_path, trust_remote_code=True, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, enable_lora=(args.lora_path is not None),
              max_lora_rank=args.lora_rank if args.lora_path else 16, dtype="bfloat16")
    lora_request = LoRARequest(args.model_tag, 1, args.lora_path) if args.lora_path else None
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    print(f"[{args.benchmark}] generating {len(all_records)} window-prompts ...")
    t0 = time.time()
    outputs = llm.generate([r["prompt"] for r in all_records], sampling_params, lora_request=lora_request)
    print(f"[{args.benchmark}] generation done in {time.time() - t0:.0f}s")

    qid_windows: Dict[str, List[Dict]] = defaultdict(list)
    for rec, output in zip(all_records, outputs):
        full_text = output.outputs[0].text.strip()
        thinking, answer_text = parse_thinking_and_answer(full_text)
        ranked_raw = parse_ranked_tables(answer_text) or parse_ranked_tables(full_text)
        qid_windows[rec["qid"]].append({
            "window_idx": rec["window_idx"], "window_start": rec["window_start"], "window_end": rec["window_end"],
            "window_candidate_ids": rec["window_candidate_ids"], "ranked_indices": ranked_raw,
            "parse_failed": ranked_raw is None, "thinking": thinking, "answer_text": answer_text,
            "prompt": rec["prompt"], "raw_output": full_text,
        })

    rows, pq_metrics, n_parse_failed = [], [], 0
    for qid, win_list in qid_windows.items():
        meta = query_meta[qid]
        gold_ids = set(meta["gold_ids"])
        win_list.sort(key=lambda w: w["window_idx"])
        final_ranking, any_failed = aggregate_bubble_sort(meta["pool"], win_list)

        if any_failed:
            n_parse_failed += 1
            m = {f"{name}@{k}": 0.0 for name in ("recall", "precision", "ndcg", "perfect_recall") for k in EVAL_K}
            m["mrr"] = 0.0
        else:
            m = compute_ranking_metrics(final_ranking, gold_ids, EVAL_K)
        pq_metrics.append(m)

        rows.append({
            "qid": qid, "benchmark": args.benchmark, "question": meta["question"],
            "gold_doc_ids": list(gold_ids), "final_ranked_list": final_ranking,
            "rank_of_gold_after_reranking": gold_ranks(final_ranking, gold_ids),
            "any_parse_failed": any_failed, "n_windows": len(win_list),
            "windows": [{"window_idx": w["window_idx"], "prompt": w["prompt"], "raw_output": w["raw_output"],
                         "thinking": w["thinking"], "answer_text": w["answer_text"],
                         "ranked_indices": w["ranked_indices"], "parse_failed": w["parse_failed"]}
                        for w in win_list],
            "metrics": m, "model_tag": args.model_tag,
        })

    out_path = os.path.join(args.out_dir, f"{args.benchmark}.jsonl")
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    agg = {}
    if pq_metrics:
        for key in pq_metrics[0]:
            agg[key] = round(sum(m[key] for m in pq_metrics) / len(pq_metrics), 4)
    agg["n_queries"] = len(rows)
    agg["n_parse_failed_questions"] = n_parse_failed
    with open(os.path.join(args.out_dir, f"{args.benchmark}_summary.json"), "w") as f:
        json.dump(agg, f, indent=2)

    print(f"[{args.benchmark}] nDCG@10={agg.get('ndcg@10', 0):.4f}  MRR={agg.get('mrr', 0):.4f}  "
          f"parse_failed={n_parse_failed}/{len(rows)}")


if __name__ == "__main__":
    main()
