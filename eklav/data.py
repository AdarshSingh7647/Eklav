"""
Training-data construction shared by all tasks (paper Table 1, Sec. 3.3).

A task describes itself with a `DataSpec`: how to read its raw corpus into
(x, r, a) examples, its hint h(r), and how a hint is worded in the prompt.
`serialize` turns an example into the (prompt, response) pair a method trains on:

  STD-CoT / MASK-ONLY   prompt = x            response = <think> r </think> a
  NAIVE-SFT             prompt = x            response = a
  EKLAV                 prompt = x + h(r)     response = a

Responses that carry reasoning wrap it in <think>...</think>, the format the
students emit at evaluation. A response without reasoning is the bare answer;
the chat template then inserts an empty reasoning block, which the answer-only
loss mask removes from the loss (see eklav.methods).

Built data is laid out as <data_root>/<task>/<layout>/{train,val}.json in
LLaMA-Factory's sharegpt format. MASK-ONLY reuses the STD-CoT layout.
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from eklav.chunks import chunk_dir, join_chunks
from eklav.methods import EKLAV, NAIVE_SFT, SEQUENCE_LAYOUT, STD_COT

Example = Dict[str, str]                  # {"x": input, "r": teacher trace, "a": answer}
Hint = Callable[[str], str]               # h(r): prompt-side hint derived from the trace
HintPrompt = Callable[[str, str], str]    # (x, h) -> prompt

# Wording that frames a teacher-derived hint in the prompt (training only; at
# evaluation the prompt is the bare task input).
HINT_INSTRUCTION = (
    "Think step-by-step before answering; here is an example of the kind of "
    "reasoning to learn from:\n"
)


@dataclass(frozen=True)
class DataSpec:
    task: str
    system: str
    val_fraction: float
    load_corpus: Callable[[str], List[Example]]
    hint: Hint
    hint_prompt: HintPrompt


def reasoning_response(reasoning: str, answer: str) -> str:
    return f"<think>\n{reasoning}\n</think>\n\n{answer}"


def serialize(layout: str, example: Example, hint: Hint, hint_prompt: HintPrompt) -> Tuple[str, str]:
    """(prompt, response) for one example under a sequence layout."""
    x, r, a = example["x"], example["r"], example["a"]
    if layout == STD_COT:
        return x, reasoning_response(r, a)
    if layout == NAIVE_SFT:
        return x, a
    if layout == EKLAV:
        return hint_prompt(x, hint(r)), a
    raise ValueError(f"Unknown sequence layout: {layout!r}")


def sharegpt_row(task: str, system: str, prompt: str, response: str) -> Dict[str, Any]:
    return {
        "system": system,
        "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": response}],
        "task": task,
    }


def val_indices(n: int, val_fraction: float, seed: int = 42) -> Set[int]:
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_val = max(1, int(n * val_fraction)) if n > 20 else 0
    return set(indices[:n_val])


def build_split(spec: DataSpec, layout: str, examples: Iterable[Example], seed: int = 42):
    """Serialize every example and split into (train_rows, val_rows). The split
    depends only on the example count and seed, so every layout of a task holds
    out the same examples."""
    examples = list(examples)
    val_idx = val_indices(len(examples), spec.val_fraction, seed)
    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(examples):
        prompt, response = serialize(layout, ex, spec.hint, spec.hint_prompt)
        row = sharegpt_row(spec.task, spec.system, prompt, response)
        (val_rows if i in val_idx else train_rows).append(row)
    return train_rows, val_rows


def layout_dir(data_root: str, task: str, layout: str) -> Path:
    return Path(data_root) / task / layout


def build_data(spec: DataSpec, layout: str, data_root: str, raw_path: str) -> Dict[str, str]:
    train_rows, val_rows = build_split(spec, layout, spec.load_corpus(raw_path))
    out_dir = layout_dir(data_root, spec.task, layout)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"train": out_dir / "train.json", "val": out_dir / "val.json"}
    for split, rows in (("train", train_rows), ("val", val_rows)):
        with open(paths[split], "w") as f:
            json.dump(rows, f)
    return {k: str(v) for k, v in paths.items()}


def get_data(spec: DataSpec, method: str, data_root: str, raw_path: Optional[str] = None) -> Dict[str, str]:
    """Paths to the method's train/val files. Reused if built; reassembled from
    `train.json.chunks/` if only the chunks are present; otherwise built from the
    raw corpus at `raw_path`."""
    layout = SEQUENCE_LAYOUT[method]
    directory = layout_dir(data_root, spec.task, layout)
    train, val = directory / "train.json", directory / "val.json"
    if not train.exists() and os.path.isdir(chunk_dir(str(train))) and val.exists():
        join_chunks(chunk_dir(str(train)), str(train))
    if train.exists() and val.exists():
        return {"train": str(train), "val": str(val)}
    if raw_path is None:
        raise ValueError(
            f"No training data for {spec.task}/{layout} under {data_root}. Pass --raw_path with the raw "
            "corpus to build it."
        )
    return build_data(spec, layout, data_root, raw_path)
