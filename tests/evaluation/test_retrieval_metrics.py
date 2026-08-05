from __future__ import annotations

import math

import pytest

from editorial_team.evaluation.retrieval_metrics import aggregate_metrics, metrics_at_k


def test_perfect_ranking_and_binary_discount_formula() -> None:
    value = metrics_at_k(("a", "b"), frozenset({"a", "b"}), 2)
    assert value.hit_rate == value.precision == value.recall == value.mrr == value.ndcg == 1.0
    expected_dcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert expected_dcg == pytest.approx(1.6309297536)


def test_no_relevant_result_and_empty_predictions() -> None:
    assert metrics_at_k(("x",), frozenset({"a"}), 3).__dict__ == {
        "hit_rate": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
    }
    assert metrics_at_k((), frozenset({"a"}), 3).precision == 0.0


def test_rank_two_mrr_and_reversed_binary_ndcg() -> None:
    value = metrics_at_k(("x", "a"), frozenset({"a"}), 2)
    assert value.mrr == 0.5
    assert value.ndcg == pytest.approx(1 / math.log2(3))


def test_several_relevant_results_and_requested_k_denominator() -> None:
    value = metrics_at_k(("a", "x", "b"), frozenset({"a", "b", "c"}), 5)
    assert value.precision == 2 / 5
    assert value.recall == 2 / 3
    assert value.hit_rate == 1.0


def test_mrr_is_truncated_at_k_and_unknown_ids_are_nonrelevant() -> None:
    predictions = ("unknown", "also-unknown", "a")
    assert metrics_at_k(predictions, frozenset({"a"}), 2).mrr == 0.0
    assert metrics_at_k(predictions, frozenset({"a"}), 3).mrr == 1 / 3


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k_is_rejected(k: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        metrics_at_k(("a",), frozenset({"a"}), k)  # type: ignore[arg-type]


def test_duplicate_predictions_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        metrics_at_k(("a", "a"), frozenset({"a"}), 2)


def test_empty_golden_is_excluded_from_deterministic_aggregation() -> None:
    cases = (
        ("case-a", ("a",), frozenset({"a"})),
        ("case-b", ("noise",), frozenset()),
        ("case-c", ("x", "c"), frozenset({"c"})),
    )
    values, empty = aggregate_metrics(cases, (1, 3))
    assert empty == 1
    assert values[1].hit_rate == 0.5
    assert values[3].mrr == 0.75
    assert list(values) == [1, 3]


def test_aggregation_requires_stable_unique_case_order() -> None:
    with pytest.raises(ValueError, match="ordered"):
        aggregate_metrics(
            (("case-b", ("b",), frozenset({"b"})), ("case-a", ("a",), frozenset({"a"}))),
            (1,),
        )
    with pytest.raises(ValueError, match="unique"):
        aggregate_metrics(
            (("case-a", ("a",), frozenset({"a"})), ("case-a", ("a",), frozenset({"a"}))),
            (1,),
        )
