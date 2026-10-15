"""
The training conditions compared in the paper (Table 1, Sec. 3.3), under the
paper's names. Data builders, task definitions and the launcher all take their
method vocabulary from here.

Notation: an example is (x, r, a) -- task input, teacher reasoning trace, final
answer. A task defines a hint h(r), the part of the trace placed in the prompt.
The student's own reasoning is never part of a training sequence: at evaluation
all teacher context is removed and the student generates its own reasoning
before answering.

  NAIVE-SFT  y = [x;           a]       loss on a          no teacher reasoning at all
  STD-CoT    y = [x;      r + a]        loss on r and a    the trace is the imitation target
  EKLAV      y = [x + h(r);    a]       loss on a          the trace is prompt-side context
  MASK-ONLY  y = [x;      r + a]        loss on a          ablation: no hint; the trace sits in
                                                           the response with its loss masked

EKLAV is cheaper to train than STD-CoT when its hint is shorter than the trace:
its sequence then holds a reduced trace instead of the whole trace plus answer.
Masking loss does not by itself reduce the FLOPs spent on a sequence. MASK-ONLY
trains on exactly STD-CoT's sequence and differs only in the loss mask, which
isolates masking from EKLAV's hint.
"""

from typing import Dict

STD_COT = "std-cot"
NAIVE_SFT = "naive-sft"
EKLAV = "eklav"
MASK_ONLY = "mask-only"

METHODS = (STD_COT, NAIVE_SFT, EKLAV, MASK_ONLY)

PAPER_NAMES = {STD_COT: "STD-CoT", NAIVE_SFT: "NAIVE-SFT", EKLAV: "EKLAV", MASK_ONLY: "MASK-ONLY"}

# Which serialized sequence a method trains on. MASK-ONLY shares STD-CoT's.
SEQUENCE_LAYOUT = {STD_COT: STD_COT, MASK_ONLY: STD_COT, NAIVE_SFT: NAIVE_SFT, EKLAV: EKLAV}

# Every answer-only objective (NAIVE-SFT, EKLAV, MASK-ONLY) masks everything up
# to and including the closing reasoning tag. That also covers the empty
# reasoning block the chat template inserts into a response that has none
# (paper Appendix A.3). STD-CoT supervises the whole response.
ANSWER_ONLY_MASK_ENV = {"EKLAV_ANSWER_ONLY_AFTER_TOKEN": "</think>"}


def loss_mask_env(method: str) -> Dict[str, str]:
    """Environment variables for the masking plugin that realize `method`'s loss mask."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
    return {} if method == STD_COT else dict(ANSWER_ONLY_MASK_ENV)
