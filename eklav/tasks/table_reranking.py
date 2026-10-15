"""
Table reranking (listwise JSON ranking), trained on a corpus of DeepSeek-R1
traces (sft_examples.json: qid / prompt / thinking / answer).

Hint h(r) = r. Truncating a multi-candidate comparison would remove candidates
from consideration rather than just stop an argument early, so the complete
trace goes in the prompt and the response is the bare answer JSON. The trace is
only ~14% of a table sequence, so unlike passage reranking this layout is not
shorter than STD-CoT's.

Hyperparameters (paper Sec. 4): the passage recipe -- LoRA rank 32 / alpha 64 on
all linear layers, lr 1e-4, cosine schedule with 5% warmup, effective batch 128,
bf16, two epochs -- with the sequence cutoff raised to 32,768 tokens. The
per-device batch is 1 because a single 32,768-token sequence already takes ~80 GB
on an H200; gradient accumulation is derived from it so the effective batch
stays 128.
"""

import json
import re
from typing import Any, Dict, List, Optional

from eklav.data import HINT_INSTRUCTION, DataSpec, Example
from eklav.tasks.base import Task

NAME = "table_reranking"

SYSTEM_PROMPT = (
    "You are a careful retrieval assistant that ranks candidate tables by "
    "their relevance to a user's question."
)

REQUIRED_RAW_FIELDS = ("qid", "prompt", "thinking", "answer")


def _extract_answer_json(answer: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", answer, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _normalize_answer(answer: str) -> str:
    parsed = _extract_answer_json(answer)
    return json.dumps(parsed) if parsed is not None else answer.strip()


def _is_valid(ex: Dict[str, Any]) -> bool:
    for field in REQUIRED_RAW_FIELDS:
        if field not in ex or ex[field] is None:
            return False
        if isinstance(ex[field], str) and not ex[field].strip():
            return False
    parsed = _extract_answer_json(ex["answer"])
    if parsed is None:
        return False
    ranked = parsed.get("ranked_tables")
    return isinstance(ranked, list) and len(ranked) > 0


def load_corpus(raw_path: str) -> List[Example]:
    """sft_examples.json -> (x, r, a) examples, dropping malformed rows and duplicate qids."""
    with open(raw_path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("examples", []))

    examples, seen = [], set()
    for ex in raw:
        if not _is_valid(ex) or ex["qid"] in seen:
            continue
        seen.add(ex["qid"])
        examples.append({
            "x": ex["prompt"].strip(),
            "r": ex["thinking"].strip(),
            "a": _normalize_answer(ex["answer"]),
        })
    return examples


def hint(trace: str) -> str:
    """h(r) = r: the whole trace."""
    return trace


def hint_prompt(x: str, h: str) -> str:
    return f"{x}\n\n{HINT_INSTRUCTION}{h}"


TASK = Task(
    name=NAME,
    data=DataSpec(
        task=NAME, system=SYSTEM_PROMPT, val_fraction=0.05,
        load_corpus=load_corpus, hint=hint, hint_prompt=hint_prompt,
    ),
    hparams={
        "template": "qwen3",
        "enable_thinking": True,
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_target": "all",
        "learning_rate": 1.0e-4,
        "num_train_epochs": 2.0,
        "effective_batch_size": 128,
        "cutoff_len": 32768,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "per_device_train_batch_size": 1,
        "save_steps": 50,
        "save_total_limit": 4,
    },
    raw_help="sft_examples.json: a list of {qid, prompt, thinking, answer}.",
)
