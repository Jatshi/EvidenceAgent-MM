from __future__ import annotations

from evidenceagent_mm.evaluation import (
    brier_score,
    citation_temporal_iou,
    expected_calibration_error,
    retrieval_metrics,
    word_error_rate,
)


def test_retrieval_metrics() -> None:
    metrics = retrieval_metrics(["a", "b", "c"], {"b", "x"}, k=3)
    assert metrics["recall_at_k"] == 0.5
    assert metrics["mrr"] == 0.5


def test_temporal_iou() -> None:
    assert citation_temporal_iou((100, 300), (200, 400)) == 1 / 3
    assert citation_temporal_iou((0, 10), (20, 30)) == 0.0


def test_calibration_metrics_are_bounded() -> None:
    probabilities = [0.1, 0.8, 0.9]
    labels = [0, 1, 1]
    assert 0 <= brier_score(probabilities, labels) <= 1
    assert 0 <= expected_calibration_error(probabilities, labels, bins=3) <= 1


def test_word_error_rate_normalizes_case_and_punctuation() -> None:
    assert word_error_rate("Review the budget.", "review the budget") == 0.0
    assert word_error_rate("review the budget", "renew the budget") == 1 / 3
