"""
Math reasoning data: download (Eklav, std-SFT from HF) + build (Answer-only, local).

HF repos (already built by the author):
  AdarshSingh7647/Eklav-Math-Data        -- Eklav method, train.json/val.json
  AdarshSingh7647/Eklav-Math-CotGen-Data -- std-SFT method, train.json/val.json

Answer-only has no HF repo yet, so build_answer_only() builds it locally,
applying the same "no hint, no reasoning, loss on answer only" principle as
passage reranking's Answer-only, from the same stage2-3k raw source the
Eklav/std-SFT HF repos were built from.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from huggingface_hub import hf_hub_download

HF_REPOS = {
    "eklav": "AdarshSingh7647/Eklav-Math-Data",
    "std-sft": "AdarshSingh7647/Eklav-Math-CotGen-Data",
}

SYSTEM_PROMPT = (
    "You are a careful math assistant that reasons step-by-step and reports a "
    "final boxed answer."
)

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>\s*(.*)", flags=re.DOTALL)


def _data_dir(root: str) -> Path:
    d = Path(root) / "math_reasoning"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_hf_method(method: str, data_root: str) -> Dict[str, str]:
    if method not in HF_REPOS:
        raise ValueError(f"No HF repo for math_reasoning/{method}; use build_answer_only() instead.")
    repo_id = HF_REPOS[method]
    out_dir = _data_dir(data_root) / method
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for split in ("train", "val"):
        local_path = out_dir / f"{split}.json"
        if local_path.exists():
            paths[split] = str(local_path)
            continue
        downloaded = hf_hub_download(repo_id=repo_id, filename=f"{split}.json", repo_type="dataset")
        local_path.write_bytes(Path(downloaded).read_bytes())
        paths[split] = str(local_path)
    return paths


def parse_output(assistant_content: str) -> Optional[Dict[str, str]]:
    m = THINK_RE.search(assistant_content)
    if not m:
        return None
    thinking = m.group(1).strip()
    answer = m.group(2).strip()
    if not thinking or not answer or "\\boxed" not in answer:
        return None
    return {"thinking": thinking, "answer": answer}


def _make_conversation(turns: List[Dict[str, str]]) -> Dict[str, Any]:
    return {"system": SYSTEM_PROMPT, "conversations": turns, "task": "math_reasoning"}


def build_answer_only(raw_dataset_path: str, data_root: str, val_fraction: float = 0.01,
                       seed: int = 42) -> Dict[str, str]:
    """Build Answer-only SFT sharegpt data locally: human=problem (no hint),
    gpt=bare worked-solution answer only (no <think> block at all), analogous
    to passage reranking's naive() builder. `raw_dataset_path` is the
    stage2-3k.json file (list of {"conversations": [user, assistant]} rows,
    assistant = "<think>...</think>\\n\\n<solution ending in \\boxed{}>")."""
    import random

    out_dir = _data_dir(data_root) / "answer-only"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(raw_dataset_path) as f:
        rows = json.load(f)

    examples = []
    for row in rows:
        convs = row.get("conversations", [])
        if len(convs) != 2 or convs[0]["from"] != "user" or convs[1]["from"] != "assistant":
            continue
        parsed = parse_output(convs[1]["value"])
        if parsed is None:
            continue
        examples.append({"prompt": convs[0]["value"].strip(), "answer": parsed["answer"]})

    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    n_val = max(1, int(len(examples) * val_fraction))
    val_idx = set(indices[:n_val])

    train_rows, val_rows = [], []
    for i, ex in enumerate(examples):
        row = _make_conversation([
            {"from": "human", "value": ex["prompt"]},
            {"from": "gpt", "value": ex["answer"]},
        ])
        (val_rows if i in val_idx else train_rows).append(row)

    train_path = out_dir / "train.json"
    val_path = out_dir / "val.json"
    with open(train_path, "w") as f:
        json.dump(train_rows, f)
    with open(val_path, "w") as f:
        json.dump(val_rows, f)

    return {"train": str(train_path), "val": str(val_path)}


def get_data(method: str, data_root: str, raw_dataset_path: Optional[str] = None) -> Dict[str, str]:
    if method in HF_REPOS:
        return download_hf_method(method, data_root)
    if method == "answer-only":
        existing_train = _data_dir(data_root) / "answer-only" / "train.json"
        existing_val = _data_dir(data_root) / "answer-only" / "val.json"
        if existing_train.exists() and existing_val.exists():
            return {"train": str(existing_train), "val": str(existing_val)}
        if raw_dataset_path is None:
            raise ValueError(
                "math_reasoning/answer-only has no HF repo yet; pass --raw_dataset_path "
                "pointing at stage2-3k.json to build it locally."
            )
        return build_answer_only(raw_dataset_path, data_root)
    raise ValueError(f"Unknown method for math_reasoning: {method}")
