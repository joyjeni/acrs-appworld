"""Ranking, routing, consensus and compression metrics.

Implemented from scratch (no scikit-learn / pytrec_eval) so the same numbers
can be reproduced in an offline Kaggle kernel.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    hit = len(set(ranked[:k]) & gold_set)
    return hit / min(len(gold_set), k) if k < len(gold_set) else hit / len(gold_set)


def true_recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Unclipped recall: |top-k ∩ gold| / |gold|."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return len(set(ranked[:k]) & gold_set) / len(gold_set)


def precision_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    gold_set = set(gold)
    return len(set(ranked[:k]) & gold_set) / k


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    gains = [1.0 if d in gold_set else 0.0 for d in ranked[:k]]
    ideal = [1.0] * min(len(gold_set), k)
    denom = dcg(ideal)
    return dcg(gains) / denom if denom else 0.0


def mrr(ranked: Sequence[str], gold: Iterable[str]) -> float:
    gold_set = set(gold)
    for i, d in enumerate(ranked):
        if d in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(ranked: Sequence[str], gold: Iterable[str]) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    hits = 0
    total = 0.0
    for i, d in enumerate(ranked):
        if d in gold_set:
            hits += 1
            total += hits / (i + 1)
    return total / len(gold_set)


def f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def set_f1(predicted: Iterable[str], gold: Iterable[str]) -> float:
    pred, g = set(predicted), set(gold)
    if not pred or not g:
        return 0.0
    tp = len(pred & g)
    p = tp / len(pred)
    r = tp / len(g)
    return f1(p, r)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def stdev(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def summarise(name: str, values: Sequence[float]) -> dict[str, float]:
    return {
        f"{name}_mean": round(mean(values), 6),
        f"{name}_std": round(stdev(values), 6),
        f"{name}_n": float(len(values)),
    }


def compression_ratio(kept_tokens: int, original_tokens: int) -> float:
    if original_tokens <= 0:
        return 0.0
    return 1.0 - kept_tokens / original_tokens


__all__ = [
    "average_precision",
    "compression_ratio",
    "dcg",
    "f1",
    "mean",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "set_f1",
    "stdev",
    "summarise",
    "true_recall_at_k",
]
