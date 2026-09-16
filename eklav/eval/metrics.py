"""
Shared eval metrics/helpers, ported 1:1 from the reference implementations:
  - Brace-balanced \\boxed{} extraction + math_verify scoring (math reasoning).
  - MCQ letter extraction (math reasoning: GPQA-Diamond, MMLU).
  - nDCG@k / recall@k / precision@k / MRR (table reranking; also usable for
    any binary-relevance ranking).
"""

import math
import re
from typing import Dict, List, Optional, Sequence, Set

# --------------------------------------------------------------------------- #
# Math reasoning: boxed-answer extraction + scoring
# --------------------------------------------------------------------------- #

_BOXED_KEY = "\\boxed{"
_LETTER_RE = re.compile(r"[A-Da-d]")


def extract_boxed(text: str) -> Optional[str]:
    """Brace-balanced extraction of the LAST \\boxed{...} in `text`. Returns
    None if unbalanced or not found. A naive non-greedy regex breaks on
    nested braces (e.g. \\boxed{\\frac{1}{2}})."""
    idx = text.rfind(_BOXED_KEY)
    if idx == -1:
        return None
    start = idx + len(_BOXED_KEY)
    depth, i, n = 1, start, len(text)
    while i < n and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start:i - 1]


def normalize_ws(s: Optional[str]) -> Optional[str]:
    return re.sub(r"\s+", "", s) if s is not None else None


def extract_mcq_letter(text: str) -> Optional[str]:
    boxed = extract_boxed(text)
    candidate = boxed if boxed is not None else text[-40:]
    m = _LETTER_RE.search(candidate)
    return m.group(0).upper() if m else None


_MATH_VERIFY_UNAVAILABLE = False


def math_answers_equal(gold: str, pred_boxed: Optional[str]) -> bool:
    """Symbolic/numeric equivalence via math-verify, falling back to
    whitespace-normalized exact string match if math-verify errors out."""
    global _MATH_VERIFY_UNAVAILABLE
    if pred_boxed is None:
        return False
    if normalize_ws(gold) == normalize_ws(pred_boxed):
        return True
    if _MATH_VERIFY_UNAVAILABLE:
        return False
    try:
        from math_verify import parse, verify

        gold_parsed = parse(f"\\boxed{{{gold}}}")
        pred_parsed = parse(f"\\boxed{{{pred_boxed}}}")
        return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        return False


def mcq_answers_equal(gold_letter: str, response_text: str) -> bool:
    pred = extract_mcq_letter(response_text)
    return pred is not None and pred == gold_letter.upper()


# --------------------------------------------------------------------------- #
# Table reranking: ranking metrics (binary relevance)
# --------------------------------------------------------------------------- #

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
    m = {"mrr": round(mrr(ranked_ids, gold_ids), 4)}
    for k in eval_k:
        m[f"recall@{k}"] = round(recall_at_k(ranked_ids, gold_ids, k), 4)
        m[f"precision@{k}"] = round(precision_at_k(ranked_ids, gold_ids, k), 4)
        m[f"ndcg@{k}"] = round(ndcg_at_k(ranked_ids, gold_ids, k), 4)
        m[f"perfect_recall@{k}"] = round(perfect_recall_at_k(ranked_ids, gold_ids, k), 4)
    return m
