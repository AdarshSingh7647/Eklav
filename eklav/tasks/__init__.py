"""Task registry. To add a task, write a module that defines a `TASK` and list it here."""

from typing import Dict

from eklav.tasks import passage_reranking, table_reranking
from eklav.tasks.base import Task

TASKS: Dict[str, Task] = {t.name: t for t in (passage_reranking.TASK, table_reranking.TASK)}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise ValueError(f"Unknown task {name!r}; available: {sorted(TASKS)}")
    return TASKS[name]
