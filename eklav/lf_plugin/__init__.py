"""LLaMA-Factory integration: the answer-only loss mask (see `masking`)."""

from pathlib import Path

from eklav.lf_plugin.masking import apply

__all__ = ["apply", "startup_dir"]


def startup_dir() -> str:
    """Directory holding a `sitecustomize.py` that applies the mask at interpreter
    startup. Put it on PYTHONPATH of a `llamafactory-cli` launch so that every
    process -- including the workers torchrun spawns for multi-GPU training --
    carries the patch."""
    return str(Path(__file__).resolve().parent / "startup")
