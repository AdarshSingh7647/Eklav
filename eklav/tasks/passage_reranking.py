"""
Passage reranking (pointwise true/false), trained on the public Rank1 corpus
(jhu-clsp/rank1-training-data; Weller et al., 2025).

Hint h(r). Write the teacher trace as sentences r = (r_1, ..., r_m). The hint is
the first half of the trace with verdict-bearing sentences removed:

    h(r) = F(r_1 ... r_k),   k = max(1, round(m * 0.5))

F drops every sentence matching VERDICT_RE, the executable form of the lexical
filter. The hint is therefore shorter than the trace twice over: the second half
is never shown, and verdict-bearing sentences are dropped from the half that is.
The hint goes in the prompt and the response is the bare answer. (If F empties
the prefix the first sentence is used instead; this does not occur on the Rank1
corpus.)

Hyperparameters (paper Sec. 4): LoRA rank 32 / alpha 64 on all linear layers,
lr 1e-4, cosine schedule with 5% warmup, effective batch 128, bf16, two epochs,
sequence cutoff 2,500 tokens. `per_device_train_batch_size` is a memory knob:
gradient accumulation is derived from it so the effective batch stays 128.
"""

import re
from typing import Dict, List, Optional

from eklav.data import HINT_INSTRUCTION, DataSpec, Example
from eklav.tasks.base import Task

NAME = "passage_reranking"

SYSTEM_PROMPT = (
    "You are a careful retrieval assistant that judges whether a passage is "
    "relevant to a user's query."
)

# Closes a hint so the prompt ends on an instruction rather than trailing reasoning.
HINT_CLOSING = "\n\nNow reason and output the final verdict."

# Fraction of the trace's sentences (by count) exposed in the hint, before filtering.
HINT_FRACTION = 0.5

THINK_RE = re.compile(r"<think>(.*?)</think>\s*(.*)", flags=re.DOTALL)

# Sentence boundary: whitespace following terminal punctuation. Every trace has
# one; ~16% of traces contain no newline, so newlines cannot delimit sentences.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# Verdict-bearing sentences assert (or hedge toward) the relevance / true-false
# judgment itself, as opposed to restating the task framing ("...if the passage
# is relevant to the query..."). The `(?!\s+to\b)` guard tells the two apart.
VERDICT_RE = re.compile(
    r"\b(ir)?relevant\b(?!\s+to\b)"
    r"|\b(is|are|be|seems?|being)\s+(true|false)\b"
    r"|\banswer\s+(is|should be)\s+(true|false)\b"
    r"|\bnot\s+directly\s+relevant\b",
    re.IGNORECASE,
)


def parse_output(output: str) -> Optional[Dict[str, str]]:
    """Split '<think>...</think> true' into {'reasoning': ..., 'answer': 'true'}."""
    m = THINK_RE.search(output)
    if not m:
        return None
    reasoning = m.group(1).strip()
    answer = m.group(2).strip().lower()
    if not reasoning or answer not in ("true", "false"):
        return None
    return {"reasoning": reasoning, "answer": answer}


def split_sentences(trace: str) -> List[str]:
    return [t for t in _SENTENCE_BOUNDARY.split(trace.strip()) if t]


def hint(trace: str) -> str:
    """h(r): the verdict-free first half of the trace's sentences."""
    sentences = split_sentences(trace)
    k = max(1, round(len(sentences) * HINT_FRACTION))
    kept = [s for s in sentences[:k] if not VERDICT_RE.search(s)]
    return " ".join(kept if kept else sentences[:1])


def hint_prompt(x: str, h: str) -> str:
    return f"{x}\n\n{HINT_INSTRUCTION}{h}{HINT_CLOSING}"


def load_corpus(raw_path: str) -> List[Example]:
    """Rank1 arrow file (instruction / input / output columns) -> (x, r, a) examples."""
    from datasets import Dataset

    examples = []
    for row in Dataset.from_file(raw_path):
        parsed = parse_output(row["output"])
        if parsed is None:
            continue
        examples.append({
            "x": f"{row['instruction'].strip()}\n\n{row['input'].strip()}",
            "r": parsed["reasoning"],
            "a": parsed["answer"],
        })
    return examples


TASK = Task(
    name=NAME,
    data=DataSpec(
        task=NAME, system=SYSTEM_PROMPT, val_fraction=0.01,
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
        "cutoff_len": 2500,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "seed": 12345,
        "per_device_train_batch_size": 64,
        "save_steps": 250,
        "save_total_limit": 2,
    },
    raw_help="Rank1 training data arrow file (jhu-clsp/rank1-training-data).",
)
