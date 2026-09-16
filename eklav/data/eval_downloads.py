"""
Download code for the eval-time benchmark sources shared across tasks:
  - BRIGHT (passage reranking eval, mteb/BRIGHT, pinned revision).
  - rank1-run-files (passage reranking eval, precomputed BM25 candidate pools).
Both public, no auth needed.
"""

import json
from typing import Dict, List, Tuple

from huggingface_hub import hf_hub_download

BRIGHT_DATASET_PATH = "mteb/BRIGHT"
BRIGHT_DATASET_REVISION = "c26703e6600d97c579ee2985f16cf307db13ed85"
BRIGHT_DOMAINS = [
    "biology", "earth_science", "economics", "psychology", "robotics",
    "stackoverflow", "sustainable_living", "pony", "leetcode", "aops",
    "theoremqa_theorems", "theoremqa_questions",
]

RANK1_RUNFILES_REPO = "jhu-clsp/rank1-run-files"


def load_bright_domain(domain: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, int]], Dict[str, set]]:
    """Returns (corpus, queries, qrels, excluded_ids) for one BRIGHT domain."""
    from datasets import load_dataset

    corpus_ds = load_dataset(BRIGHT_DATASET_PATH, "documents", split=domain, revision=BRIGHT_DATASET_REVISION)
    examples = load_dataset(BRIGHT_DATASET_PATH, "examples", split=domain, revision=BRIGHT_DATASET_REVISION)

    corpus = {e["id"]: e["content"] for e in corpus_ds}
    queries = {e["id"]: e["query"] for e in examples}
    qrels: Dict[str, Dict[str, int]] = {}
    excluded: Dict[str, set] = {}
    for e in examples:
        qid = e["id"]
        qrels[qid] = {gid: 1 for gid in e["gold_ids"]}
        excl_ids = e.get("excluded_ids") or []
        excluded[qid] = set(excl_ids) if excl_ids != ["N/A"] else set()
    return corpus, queries, qrels, excluded


def load_rank1_run_file(filename: str) -> Dict[str, Dict[str, float]]:
    """Download+parse one {query_id: {doc_id: score}} run file from
    jhu-clsp/rank1-run-files (cached by huggingface_hub's own cache)."""
    path = hf_hub_download(RANK1_RUNFILES_REPO, filename, repo_type="dataset")
    with open(path) as f:
        return json.load(f)


def bright_runfile_name(domain: str) -> str:
    return f"{domain}_bm25_long_False/score.json"


def top_k_from_run(run: Dict[str, Dict[str, float]], top_k: int) -> Dict[str, List[str]]:
    return {
        qid: [cid for cid, _ in sorted(cands.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
        for qid, cands in run.items()
    }
