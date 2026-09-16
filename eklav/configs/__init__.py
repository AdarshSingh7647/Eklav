"""
Per-task LoRA SFT hyperparameter defaults, read verbatim from the reference
config generators (see each task's module for exact provenance/citations).
"""

from . import math_reasoning, passage_reranking, table_reranking

TASK_CONFIGS = {
    "passage_reranking": passage_reranking.CONFIG,
    "table_reranking": table_reranking.CONFIG,
    "math_reasoning": math_reasoning.CONFIG,
}
