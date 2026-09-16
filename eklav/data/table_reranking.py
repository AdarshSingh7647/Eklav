"""
Table reranking data: no HF repo exists yet for any method (std-sft,
answer-only, eklav) -- build all three locally from the raw sft_examples.json
source, mirroring the reference builders exactly:
  std-sft     <- setup1_full_loss
  answer-only <- setup2_answer_only
  eklav       <- setup3_1_hint_answer_only (confirmed by the author to be the
                 exact Eklav method for table reranking)

Raw source resolution order:
  1. --raw_path if given explicitly.
  2. TABLE_RERANKING_RAW_DATA_URL, once the author provides a public HF/Dropbox
     link (placeholder -- not set yet).
  3. Local fallback at LOCAL_RAW_FALLBACK_PATH (only exists on the original
     author's machine; not portable to other machines).
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Placeholder: author will provide a public HF or Dropbox link for the raw
# table-reranking training source (sft_examples.json + metadata files) here.
# Until then this stays None and callers must supply --raw_path or rely on
# LOCAL_RAW_FALLBACK_PATH (below), which only exists on the machine this repo
# was originally built on.
TABLE_RERANKING_RAW_DATA_URL: Optional[str] = None

# Fallback local source for the machine this repo was built on. Will NOT
# exist on any other machine -- purely so this repo is immediately runnable
# here before the public link exists.
LOCAL_RAW_FALLBACK_DIR = "/mnt/data2/asing725_2/forge/model-forge/data/tabular_reranking/sharegpt_setups"

SYSTEM_PROMPT = (
    "You are a careful retrieval assistant that ranks candidate tables by "
    "their relevance to a user's question."
)

HINT_INSTRUCTION = (
    "Think step-by-step before answering; here is an example of the kind of "
    "reasoning to learn from:\n"
)

REQUIRED_RAW_FIELDS = ["qid", "prompt", "thinking", "answer"]

METHOD_TO_SETUP = {
    "std-sft": "setup1_full_loss",
    "answer-only": "setup2_answer_only",
    "eklav": "setup3_1_hint_answer_only",
}


def _data_dir(root: str) -> Path:
    d = Path(root) / "table_reranking"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _extract_answer_json(answer_str: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", answer_str, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _normalize_answer_str(answer_str: str) -> str:
    parsed = _extract_answer_json(answer_str)
    return json.dumps(parsed) if parsed is not None else answer_str.strip()


def _is_valid_example(ex: Dict[str, Any]) -> bool:
    for field in REQUIRED_RAW_FIELDS:
        if field not in ex or ex[field] is None:
            return False
        if isinstance(ex[field], str) and len(ex[field].strip()) == 0:
            return False
    answer_json = _extract_answer_json(ex["answer"])
    if answer_json is None or "ranked_tables" not in answer_json:
        return False
    if not isinstance(answer_json["ranked_tables"], list) or len(answer_json["ranked_tables"]) == 0:
        return False
    return True


def resolve_raw_path(raw_path: Optional[str] = None) -> str:
    if raw_path:
        return raw_path
    if TABLE_RERANKING_RAW_DATA_URL:
        raise NotImplementedError(
            "TABLE_RERANKING_RAW_DATA_URL is set but automatic download from it is not yet "
            "implemented -- download it manually and pass --raw_path."
        )
    if os.path.isdir(LOCAL_RAW_FALLBACK_DIR):
        raise FileNotFoundError(
            f"No raw sft_examples.json path given. {LOCAL_RAW_FALLBACK_DIR} exists but holds "
            "already-built sharegpt setups, not the raw source -- pass --raw_path explicitly, "
            "or point --sharegpt_fallback_dir at it to reuse those prebuilt files directly "
            "(see load_prebuilt_local_fallback())."
        )
    raise FileNotFoundError(
        "No raw table-reranking source available: TABLE_RERANKING_RAW_DATA_URL is not yet set "
        "(author has not published the public link), and the local fallback directory "
        f"({LOCAL_RAW_FALLBACK_DIR}) does not exist on this machine. Pass --raw_path explicitly "
        "once you have sft_examples.json."
    )


def load_and_filter_raw(raw_path: str) -> List[Dict[str, Any]]:
    with open(raw_path) as f:
        raw_data = json.load(f)
    if isinstance(raw_data, dict):
        raw_data = raw_data.get("data", raw_data.get("examples", []))

    filtered, seen_qids = [], set()
    for ex in raw_data:
        if not _is_valid_example(ex):
            continue
        qid = ex.get("qid")
        if qid is not None and qid in seen_qids:
            continue
        if qid is not None:
            seen_qids.add(qid)
        filtered.append({
            "qid": ex["qid"],
            "dataset": ex.get("dataset", "unknown"),
            "prompt": ex["prompt"].strip(),
            "thinking": ex["thinking"].strip(),
            "answer": _normalize_answer_str(ex["answer"]),
        })
    return filtered


def _make_conversation(turns: List[Dict[str, str]]) -> Dict[str, Any]:
    return {"system": SYSTEM_PROMPT, "conversations": turns, "task": "table_reranking"}


def setup1_full_loss(ex: Dict[str, Any]) -> Dict[str, Any]:
    response = f"{ex['thinking']}\n\n{ex['answer']}"
    return _make_conversation([
        {"from": "human", "value": ex["prompt"]},
        {"from": "gpt", "value": response},
    ])


def setup2_answer_only(ex: Dict[str, Any]) -> Dict[str, Any]:
    return _make_conversation([
        {"from": "human", "value": ex["prompt"]},
        {"from": "gpt", "value": ex["thinking"]},
        {"from": "human", "value": "Based on that reasoning, give the final ranking."},
        {"from": "gpt", "value": ex["answer"]},
    ])


def _prompt_with_hint(ex: Dict[str, Any]) -> str:
    return f"{ex['prompt']}\n\n{HINT_INSTRUCTION}{ex['thinking']}"


def setup3_1_hint_answer_only(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Eklav method: full untruncated teacher trace as hint in the prompt,
    bare answer JSON only in the response (no <think> block at all)."""
    return _make_conversation([
        {"from": "human", "value": _prompt_with_hint(ex)},
        {"from": "gpt", "value": ex["answer"]},
    ])


SETUP_BUILDERS = {
    "setup1_full_loss": setup1_full_loss,
    "setup2_answer_only": setup2_answer_only,
    "setup3_1_hint_answer_only": setup3_1_hint_answer_only,
}

MASK_HISTORY_REQUIRED = {"setup2_answer_only"}


def build_method(method: str, data_root: str, raw_path: Optional[str] = None,
                  val_fraction: float = 0.05, seed: int = 42) -> Dict[str, str]:
    import random

    if method not in METHOD_TO_SETUP:
        raise ValueError(f"Unknown method for table_reranking: {method}")
    setup_name = METHOD_TO_SETUP[method]
    builder = SETUP_BUILDERS[setup_name]

    out_dir = _data_dir(data_root) / method
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path = out_dir / "train.json", out_dir / "val.json"
    if train_path.exists() and val_path.exists():
        return {"train": str(train_path), "val": str(val_path)}

    resolved_raw_path = resolve_raw_path(raw_path)
    examples = load_and_filter_raw(resolved_raw_path)

    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    n_val = max(1, int(len(examples) * val_fraction)) if len(examples) > 20 else 0
    val_idx = set(indices[:n_val])

    train_rows, val_rows = [], []
    for i, ex in enumerate(examples):
        row = builder(ex)
        (val_rows if i in val_idx else train_rows).append(row)

    with open(train_path, "w") as f:
        json.dump(train_rows, f)
    with open(val_path, "w") as f:
        json.dump(val_rows, f)

    return {"train": str(train_path), "val": str(val_path)}


def get_data(method: str, data_root: str, raw_path: Optional[str] = None) -> Dict[str, str]:
    return build_method(method, data_root, raw_path=raw_path)


# --------------------------------------------------------------------------- #
# OOD eval benchmark downloads (IBM Research retrieval repos + BRIGHT reuse)
# --------------------------------------------------------------------------- #

OOD_BENCHMARK_REPOS = {
    "AITQARetrieval": "ibm-research/AITQARetrieval",
    "FeTaQARetrieval": "ibm-research/FeTaQARetrieval",
    "MultiHierttRetrieval": "ibm-research/MultiHierttRetrieval",
    "OTTQASmallRetrieval": "ibm-research/OTTQASmallRetrieval",
    "OpenWikiTablesRetrieval": "ibm-research/OpenWikiTablesRetrieval",
    "NQTablesRetrieval": "ibm-research/NQTablesRetrieval",  # in-domain, not OOD
}


def download_benchmark(name: str, cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load corpus.jsonl / test_queries.jsonl / test_qrels.jsonl (or dev_*
    for datasets without a test split) for one IBM table-retrieval benchmark
    via `datasets.load_dataset`, returning the three splits as lists of dicts."""
    from datasets import load_dataset

    if name not in OOD_BENCHMARK_REPOS:
        raise ValueError(f"Unknown table_reranking OOD benchmark: {name}")
    repo_id = OOD_BENCHMARK_REPOS[name]

    def _load_first(configs_and_splits):
        for config, split in configs_and_splits:
            try:
                return load_dataset(repo_id, config, cache_dir=cache_dir)[split]
            except Exception:
                continue
        raise RuntimeError(f"Could not load any of {configs_and_splits} for {repo_id}")

    corpus = _load_first([("corpus_md", "corpus_md"), ("corpus", "corpus"), ("corpus_linearized", "corpus_linearized")])
    queries = _load_first([("test_queries", "test_queries"), ("dev_queries", "dev_queries")])
    qrels = _load_first([("default", "test"), ("default", "dev")])

    return {"corpus": corpus, "queries": queries, "qrels": qrels}
