"""The interface a task implements to plug into training and data building."""

from dataclasses import dataclass
from typing import Any, Dict

from eklav.data import DataSpec


@dataclass(frozen=True)
class Task:
    """A reranking task.

    name:      identifier used on the command line and in output paths.
    data:      how its raw corpus becomes (x, r, a) examples and a hint h(r).
    hparams:   LoRA SFT hyperparameters shared by every method of the task.
    raw_help:  what --raw_path should point at.
    """

    name: str
    data: DataSpec
    hparams: Dict[str, Any]
    raw_help: str
