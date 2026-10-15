"""Ranking metrics for binary relevance."""

import math
from typing import Dict, List, Sequence, Set


def recall_at_k(ranked: List[str], gold: Set[str], k: int) -> float:
    if not gold:
        return 0.0
    return sum(1 for d in ranked[:k] if d in gold) / len(gold)


def precision_at_k(ranked: List[str], gold: Set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for d in ranked[:k] if d in gold) / k


def ndcg_at_k(ranked: List[str], gold: Set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked: List[str], gold: Set[str]) -> float:
    for i, d in enumerate(ranked):
        if d in gold:
            return 1.0 / (i + 1)
    return 0.0


def perfect_recall_at_k(ranked: List[str], gold: Set[str], k: int) -> float:
    return 1.0 if gold and gold.issubset(set(ranked[:k])) else 0.0


def compute_ranking_metrics(ranked_ids: List[str], gold_ids: Set[str], eval_k: Sequence[int]) -> Dict[str, float]:
    metrics = {"mrr": round(mrr(ranked_ids, gold_ids), 4)}
    for k in eval_k:
        metrics[f"recall@{k}"] = round(recall_at_k(ranked_ids, gold_ids, k), 4)
        metrics[f"precision@{k}"] = round(precision_at_k(ranked_ids, gold_ids, k), 4)
        metrics[f"ndcg@{k}"] = round(ndcg_at_k(ranked_ids, gold_ids, k), 4)
        metrics[f"perfect_recall@{k}"] = round(perfect_recall_at_k(ranked_ids, gold_ids, k), 4)
    return metrics


def zero_metrics(eval_k: Sequence[int]) -> Dict[str, float]:
    """Metrics of a query whose ranking could not be produced."""
    metrics = {"mrr": 0.0}
    for k in eval_k:
        for name in ("recall", "precision", "ndcg", "perfect_recall"):
            metrics[f"{name}@{k}"] = 0.0
    return metrics


def mean_metrics(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_query:
        return {}
    return {key: round(sum(m[key] for m in per_query) / len(per_query), 4) for key in per_query[0]}
