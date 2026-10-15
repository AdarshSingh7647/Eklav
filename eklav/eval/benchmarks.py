"""
Evaluation benchmarks.

A benchmark is a name plus a loader that returns `RetrievalData` -- queries, a
corpus and relevance judgments, optionally with precomputed first-stage
candidates. Both reranking tasks evaluate on any such benchmark.

Built in: BRIGHT (12 domains) and the IBM table-retrieval benchmarks. Anything
else needs no code:

    --benchmark local:/path/to/dir     BEIR layout: corpus.jsonl, queries.jsonl, qrels/<split>.tsv
    --benchmark hf:org/name            Hub dataset with `default` (qrels), `queries` and `corpus` configs

or register a loader in Python:

    from eklav.eval.benchmarks import Benchmark, RetrievalData, register
    register(Benchmark("my-benchmark", loader=lambda subset: RetrievalData(queries, corpus, qrels)))
"""

import csv
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class RetrievalData:
    """queries: id -> text.  corpus: id -> text.  qrels: query id -> {doc id: relevance > 0}.
    candidates: optional precomputed first-stage ranking per query (best first); when
    absent the evaluator builds candidate pools itself.
    query_template: optional wrapper containing {query}, applied to every query."""

    queries: Dict[str, str]
    corpus: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]
    candidates: Optional[Dict[str, List[str]]] = None
    query_template: Optional[str] = None


@dataclass(frozen=True)
class Benchmark:
    name: str
    loader: Callable[[str], RetrievalData]       # subset name -> data
    subsets: Tuple[str, ...] = ()                # e.g. BRIGHT's domains; empty = a single subset

    def subset_names(self) -> Tuple[str, ...]:
        return self.subsets or (self.name,)


_REGISTRY: Dict[str, Benchmark] = {}


def register(benchmark: Benchmark) -> Benchmark:
    _REGISTRY[benchmark.name] = benchmark
    return benchmark


def names() -> List[str]:
    return sorted(_REGISTRY)


def resolve(spec: str, split: str = "test") -> Benchmark:
    """A registered name, `local:<dir>` or `hf:<repo id>`."""
    if spec in _REGISTRY:
        return _REGISTRY[spec]
    if spec.startswith("local:"):
        path = spec[len("local:"):]
        return Benchmark(os.path.basename(os.path.normpath(path)), lambda _subset: load_beir_dir(path, split))
    if spec.startswith("hf:"):
        repo = spec[len("hf:"):]
        return Benchmark(repo.replace("/", "__"), lambda _subset: load_hf_retrieval(repo))
    raise ValueError(f"Unknown benchmark {spec!r}. Registered: {names()}; or use local:<dir> / hf:<repo>.")


def select_queries(data: RetrievalData, n: Optional[int] = None, seed: int = 42) -> List[str]:
    """Queries that have judgments and text, all of them (sorted) or a seeded random sample of n."""
    valid = [q for q in data.qrels if q in data.queries and data.qrels[q]]
    if n is None or n >= len(valid):
        return sorted(valid)
    return sorted(random.Random(seed).sample(valid, n))


def _positive(qrels: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    return {q: {d: s for d, s in docs.items() if s > 0} for q, docs in qrels.items()}


def _doc_text(title: str, text: str) -> str:
    title, text = (title or "").strip(), (text or "").strip()
    return f"{title}\n{text}" if title and text else (title or text)


# --------------------------------------------------------------------------- #
# BEIR-layout directories
# --------------------------------------------------------------------------- #

def load_beir_dir(path: str, split: str = "test") -> RetrievalData:
    """corpus.jsonl ({_id, title, text}), queries.jsonl ({_id, text}) and
    qrels/<split>.tsv (query-id, corpus-id, score; header row optional)."""
    corpus, queries = {}, {}
    with open(os.path.join(path, "corpus.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            corpus[str(row["_id"])] = _doc_text(row.get("title"), row.get("text"))
    with open(os.path.join(path, "queries.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            queries[str(row["_id"])] = row["text"]
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(os.path.join(path, "qrels", f"{split}.tsv")) as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 3 or not row[2].lstrip("-").isdigit():
                continue                                   # header or malformed row
            qrels[row[0]][row[1]] = int(row[2])
    return RetrievalData(queries, corpus, _positive(dict(qrels)))


# --------------------------------------------------------------------------- #
# Hub datasets with `default` (qrels) / `queries` / `corpus` configs (IBM layout)
# --------------------------------------------------------------------------- #

def _first_available(load, attempts):
    for config, split in attempts:
        try:
            return load(config)[split]
        except Exception:
            continue
    raise RuntimeError(f"None of {attempts} could be loaded")


def load_hf_retrieval(repo_id: str) -> RetrievalData:
    from datasets import load_dataset

    load = lambda config: load_dataset(repo_id, config)
    qrels_ds = _first_available(load, [("default", s) for s in ("test", "dev", "train")])
    queries_ds = _first_available(load, [("queries", s) for s in ("test_queries", "dev_queries", "train_queries")])
    corpus_ds = _first_available(load, [(c, c) for c in ("corpus_md", "corpus_linearized", "corpus")])

    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for row in qrels_ds:
        qrels[str(row["qid"])][str(row["did"])] = int(row["score"])
    queries = {str(r["_id"]): r["text"] for r in queries_ds}
    corpus = {str(r["_id"]): _doc_text(r.get("title"), r.get("text")) for r in corpus_ds}
    return RetrievalData(queries, corpus, _positive(dict(qrels)))


IBM_TABLE_BENCHMARKS = {
    "AITQARetrieval": "ibm-research/AITQARetrieval",
    "FeTaQARetrieval": "ibm-research/FeTaQARetrieval",
    "OTTQASmallRetrieval": "ibm-research/OTTQASmallRetrieval",
    "OpenWikiTablesRetrieval": "ibm-research/OpenWikiTablesRetrieval",
    "NQTablesRetrieval": "ibm-research/NQTablesRetrieval",
}

for _name, _repo in IBM_TABLE_BENCHMARKS.items():
    register(Benchmark(_name, loader=lambda _subset, _repo=_repo: load_hf_retrieval(_repo)))


# --------------------------------------------------------------------------- #
# BRIGHT: reasoning-intensive retrieval, scored on Rank1's released BM25 top-100 pools
# --------------------------------------------------------------------------- #

BRIGHT_DATASET = "mteb/BRIGHT"
BRIGHT_REVISION = "c26703e6600d97c579ee2985f16cf307db13ed85"
BRIGHT_DOMAINS = (
    "biology", "earth_science", "economics", "psychology", "robotics",
    "stackoverflow", "sustainable_living", "pony", "leetcode", "aops",
    "theoremqa_theorems", "theoremqa_questions",
)
RANK1_RUNFILES_REPO = "jhu-clsp/rank1-run-files"

_BRIGHT_DEFAULT = (
    "Can you find background information about the concepts used to answer the question:\n\n"
    "{query}\n\n"
    "A passage is relevant if it contains background information about a **sub-concept** that "
    "someone might cite/link to when answering the above question."
)
_BRIGHT_THEOREMQA = "Find a passage which uses the same mathematical process as this one: {query}"
BRIGHT_QUERY_TEMPLATES = {
    "aops": (
        "Find different but similar math problems to {query}\n\n"
        "A document is relevant if it uses the same class of functions and shares **any** overlapping techniques."
    ),
    "theoremqa_questions": _BRIGHT_THEOREMQA,
    "theoremqa_theorems": _BRIGHT_THEOREMQA,
    "leetcode": (
        "I am looking to find different problems that share similar data structures (of any kind) or "
        "algorithms (e.g. DFS, DP, sorting, traversals, etc.). I am looking for problems that share one "
        "or both of these similarities to this:\n\n{query}\n\n"
        "Does this passage share any similarities? e.g. if there was a textbook on leetcode problems, "
        "this would be in the same book even though it could be in a different chapter.\n\n\n"
    ),
    "pony": (
        "I will use the programming language pony. Problem: {query}\n\n"
        "But to solve the problem above, I need to know things about pony. A passage is relevant if it "
        "contains docs that match **any** part (even basic parts) of the code I will have to write for "
        "the above program."
    ),
}


def load_bright(domain: str) -> RetrievalData:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    documents = load_dataset(BRIGHT_DATASET, "documents", split=domain, revision=BRIGHT_REVISION)
    examples = load_dataset(BRIGHT_DATASET, "examples", split=domain, revision=BRIGHT_REVISION)
    corpus = {e["id"]: e["content"] for e in documents}
    queries = {e["id"]: e["query"] for e in examples}
    qrels = {e["id"]: {gid: 1 for gid in e["gold_ids"]} for e in examples}
    excluded = {}
    for e in examples:
        ids = e.get("excluded_ids") or []
        excluded[e["id"]] = set() if ids == ["N/A"] else set(ids)

    with open(hf_hub_download(RANK1_RUNFILES_REPO, f"{domain}_bm25_long_False/score.json", repo_type="dataset")) as f:
        run = json.load(f)
    candidates = {
        qid: [d for d, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
              if d not in excluded.get(qid, set())]
        for qid, scores in run.items() if qid in queries
    }
    return RetrievalData(queries, corpus, qrels, candidates,
                         query_template=BRIGHT_QUERY_TEMPLATES.get(domain, _BRIGHT_DEFAULT))


register(Benchmark("bright", loader=load_bright, subsets=BRIGHT_DOMAINS))
