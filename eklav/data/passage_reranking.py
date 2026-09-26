"""
Passage reranking data: download (Eklav, std-SFT from HF) + build (Answer-only, local).

HF dataset repos for Eklav / std-SFT / Answer-only already exist and will be
released on acceptance; until then, set EKLAV_PR_HF_REPO_EKLAV,
EKLAV_PR_HF_REPO_STDSFT, and EKLAV_PR_HF_REPO_ANSWERONLY to the repo IDs, or
build Answer-only locally via build_answer_only() (pass --raw_arrow_path).

All three methods now download directly from HF. build_answer_only() is kept as
the local-build fallback (pass --raw_arrow_path) and documents the construction:
prompt (no hint, no reasoning) -> bare answer only, mirroring the reference
`naive()` builder. The published AnswerOnly repo was verified to match it:
identical example count and human prompts index-for-index against the
CotGen (std-SFT) split, with each response equal to that example's
post-`</think>` verdict.

eklav-mask-only (ablation, passage reranking only): isolates loss masking as
the single changed factor relative to std-SFT, holding context layout fixed.
It reuses the std-SFT data verbatim (full teacher trace in the assistant
turn, no prompt-side reasoning hint, no lexical filtering of verdict-bearing
sentences) and differs from std-SFT only in which tokens are supervised --
the `<think>...</think>` span is masked from the loss via
EKLAV_THINK_CONTENT_MASK (see eklav.configs.passage_reranking.METHOD_ENV)
instead of being trained on token-by-token. This isolates masking (factor b)
from Eklav's other two simultaneous changes vs. std-SFT: prompt-side context
placement and lexical filtering (factors a and c). See eklav-eval Section 5 /
review notes for the three-way confound this disentangles.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from huggingface_hub import hf_hub_download

# HF repo IDs are not hardcoded (anonymized submission); set these env vars
# to the real repo IDs, released on acceptance.
HF_REPOS = {
    "eklav": os.environ.get("EKLAV_PR_HF_REPO_EKLAV"),
    "std-sft": os.environ.get("EKLAV_PR_HF_REPO_STDSFT"),
    "answer-only": os.environ.get("EKLAV_PR_HF_REPO_ANSWERONLY"),
    # Ablation: same data as std-sft (full trace in the response, no hint,
    # no filter) -- only the loss mask (applied via METHOD_ENV) differs.
    "eklav-mask-only": os.environ.get("EKLAV_PR_HF_REPO_STDSFT"),
}

SYSTEM_PROMPT = (
    "You are a careful retrieval assistant that judges whether a passage is "
    "relevant to a user's query."
)

THINK_RE = re.compile(r"<think>(.*?)</think>\s*(.*)", flags=re.DOTALL)


def _data_dir(root: str) -> Path:
    d = Path(root) / "passage_reranking"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_hf_method(method: str, data_root: str) -> Dict[str, str]:
    """Download train.json/val.json for method in {'eklav', 'std-sft'} from
    its HF dataset repo, skipping the download if already cached locally."""
    if method not in HF_REPOS:
        raise ValueError(f"No HF repo for passage_reranking/{method}; use build_answer_only() instead.")
    repo_id = HF_REPOS[method]
    if not repo_id:
        raise ValueError(
            f"No HF repo ID configured for passage_reranking/{method}. The dataset repos will be "
            "released on acceptance; set the corresponding EKLAV_PR_HF_REPO_* env var in the "
            "meantime, or use build_answer_only() for the answer-only method."
        )
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


def parse_output(output: str) -> Optional[Dict[str, str]]:
    m = THINK_RE.search(output)
    if not m:
        return None
    thinking = m.group(1).strip()
    answer = m.group(2).strip().lower()
    if not thinking or answer not in ("true", "false"):
        return None
    return {"thinking": thinking, "answer": answer}


def build_prompt(instruction: str, input_text: str) -> str:
    return f"{instruction.strip()}\n\n{input_text.strip()}"


def _make_conversation(turns: List[Dict[str, str]]) -> Dict[str, Any]:
    return {"system": SYSTEM_PROMPT, "conversations": turns, "task": "passage_reranking"}


def build_answer_only(raw_arrow_path: str, data_root: str, val_fraction: float = 0.01,
                       seed: int = 42) -> Dict[str, str]:
    """Build Answer-only SFT sharegpt data locally: human=prompt (no hint,
    no reasoning), gpt=bare answer only. Mirrors the reference `naive()`
    builder (code/passage_reranking/build_sft_data.py) applied to
    jhu-clsp/rank1-training-data's arrow file (instruction/input/output
    columns, output = "<think>...</think> true|false")."""
    import random

    from datasets import Dataset

    out_dir = _data_dir(data_root) / "answer-only"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = Dataset.from_file(raw_arrow_path)
    examples = []
    for row in ds:
        parsed = parse_output(row["output"])
        if parsed is None:
            continue
        examples.append({
            "prompt": build_prompt(row["instruction"], row["input"]),
            "answer": parsed["answer"],
        })

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


def get_data(method: str, data_root: str, raw_arrow_path: Optional[str] = None) -> Dict[str, str]:
    if method in HF_REPOS:
        return download_hf_method(method, data_root)
    if method == "answer-only":
        existing_train = _data_dir(data_root) / "answer-only" / "train.json"
        existing_val = _data_dir(data_root) / "answer-only" / "val.json"
        if existing_train.exists() and existing_val.exists():
            return {"train": str(existing_train), "val": str(existing_val)}
        if raw_arrow_path is None:
            raise ValueError(
                "passage_reranking/answer-only has no HF repo yet; pass --raw_arrow_path "
                "pointing at jhu-clsp/rank1-training-data's arrow file to build it locally."
            )
        return build_answer_only(raw_arrow_path, data_root)
    raise ValueError(f"Unknown method for passage_reranking: {method}")
