"""Reference metrics for retrieval, citations, calibration, and selective answering."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def percentile(values: Sequence[float], value: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sequence."""

    if not values:
        raise ValueError("values must be non-empty")
    if not 0 <= value <= 100:
        raise ValueError("value must be in [0, 100]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * value / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute case/punctuation-insensitive Levenshtein WER without external packages."""

    def normalize(text: str) -> list[str]:
        return [token.strip(".,?!:;\"'()[]{}").lower() for token in text.split() if token.strip()]

    reference_words = normalize(reference)
    hypothesis_words = normalize(hypothesis)
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    previous = list(range(len(hypothesis_words) + 1))
    for row, reference_word in enumerate(reference_words, 1):
        current = [row]
        for column, hypothesis_word in enumerate(hypothesis_words, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1] / len(reference_words)


def retrieval_metrics(
    predicted_ids: Sequence[str], gold_ids: set[str], *, k: int = 5
) -> dict[str, float]:
    top = list(predicted_ids[:k])
    relevant = [index for index, evidence_id in enumerate(top, 1) if evidence_id in gold_ids]
    recall = len(set(top) & gold_ids) / len(gold_ids) if gold_ids else 1.0
    precision = len(set(top) & gold_ids) / len(top) if top else 0.0
    mrr = 1.0 / relevant[0] if relevant else 0.0
    return {"recall_at_k": recall, "precision_at_k": precision, "mrr": mrr}


def citation_temporal_iou(predicted: tuple[int, int], gold: tuple[int, int]) -> float:
    intersection = max(0, min(predicted[1], gold[1]) - max(predicted[0], gold[0]))
    union = max(predicted[1], gold[1]) - min(predicted[0], gold[0])
    return intersection / union if union else 0.0


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("probabilities and labels must have equal non-zero length")
    probability_array = np.asarray(probabilities, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.float64)
    if np.any((probability_array < 0) | (probability_array > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    return float(np.mean((probability_array - label_array) ** 2))


def expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[int], *, bins: int = 10
) -> float:
    if bins < 1:
        raise ValueError("bins must be positive")
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("probabilities and labels must have equal non-zero length")
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.float64)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = len(probabilities_array)
    ece = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (probabilities_array >= lower) & (
            probabilities_array <= upper if index == bins - 1 else probabilities_array < upper
        )
        count = int(mask.sum())
        if count:
            ece += (
                count
                / total
                * abs(float(probabilities_array[mask].mean() - labels_array[mask].mean()))
            )
    return float(ece)


def selective_metrics(
    predicted_answered: Sequence[bool], correct: Sequence[bool]
) -> dict[str, float]:
    if len(predicted_answered) != len(correct) or not correct:
        raise ValueError("inputs must have equal non-zero length")
    answered_indices = [index for index, answered in enumerate(predicted_answered) if answered]
    coverage = len(answered_indices) / len(correct)
    errors = sum(not correct[index] for index in answered_indices)
    risk = errors / len(answered_indices) if answered_indices else 0.0
    return {"coverage": coverage, "selective_risk": risk}
