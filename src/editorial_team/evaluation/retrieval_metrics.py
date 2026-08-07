"""Deterministic binary-relevance retrieval metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True)
class MetricsAtK:
    hit_rate: float
    precision: float
    recall: float
    mrr: float
    ndcg: float


def metrics_at_k(predictions: tuple[str, ...], golden: frozenset[str], k: int) -> MetricsAtK:
    """Score exact ordered unique predictions using requested-k precision denominator."""

    _validate_k(k)
    if not golden:
        raise ValueError("golden must not be empty")
    if len(set(predictions)) != len(predictions):
        raise ValueError("predictions must not contain duplicates")
    top = predictions[:k]
    relevant = tuple(item in golden for item in top)
    found = sum(relevant)
    first = next((index for index, value in enumerate(relevant, 1) if value), None)
    dcg = sum(1.0 / math.log2(index + 1) for index, value in enumerate(relevant, 1) if value)
    ideal_count = min(len(golden), k)
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_count + 1))
    return MetricsAtK(
        hit_rate=float(found > 0),
        precision=found / k,
        recall=found / len(golden),
        mrr=0.0 if first is None else 1.0 / first,
        ndcg=dcg / ideal,
    )


def aggregate_metrics(
    cases: tuple[tuple[str, tuple[str, ...], frozenset[str]], ...],
    k_values: tuple[int, ...],
) -> tuple[dict[int, MetricsAtK], int]:
    """Aggregate in stable case-ID order, excluding empty-golden cases."""

    if tuple(case[0] for case in cases) != tuple(sorted(case[0] for case in cases)):
        raise ValueError("cases must be ordered by case_id")
    if len({case[0] for case in cases}) != len(cases):
        raise ValueError("case IDs must be unique")
    for k in k_values:
        _validate_k(k)
    eligible = tuple(case for case in cases if case[2])
    empty_count = len(cases) - len(eligible)
    if not eligible:
        raise ValueError("at least one non-empty golden case is required")
    output: dict[int, MetricsAtK] = {}
    for k in k_values:
        rows = tuple(metrics_at_k(predictions, golden, k) for _, predictions, golden in eligible)
        output[k] = MetricsAtK(
            hit_rate=fmean(row.hit_rate for row in rows),
            precision=fmean(row.precision for row in rows),
            recall=fmean(row.recall for row in rows),
            mrr=fmean(row.mrr for row in rows),
            ndcg=fmean(row.ndcg for row in rows),
        )
    return output, empty_count


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
