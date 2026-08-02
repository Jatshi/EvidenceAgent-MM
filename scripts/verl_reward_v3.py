"""Custom reward entry point loaded by verl's Hydra configuration."""

from __future__ import annotations

import json
import re
from typing import Any


def _extract_last_json_object(text: str) -> dict[str, Any] | None:
    """Return the last JSON object embedded in a multi-turn rollout transcript.

    VERL's agent loop may pass tool-call text followed by the terminal answer to
    the reward function. Requiring the *entire* transcript to be JSON therefore
    collapses every reward to zero even when the final answer is valid.
    """

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    return candidates[-1] if candidates else None


def _trajectory_efficiency(text: str) -> float:
    """Small tie-breaker for GRPO groups; correctness remains dominant."""

    return max(0.0, 1.0 - min(len(text), 2048) / 2048.0)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any] | str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    del extra_info
    if data_source != "evidenceagent_mm_v3":
        return 0.0
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return 0.0
    if not isinstance(solution_str, str):
        return 0.0
    predicted = _extract_last_json_object(solution_str)
    if predicted is None:
        # Dense, auditable shaping for early rollouts that have not yet learned
        # the exact terminal JSON contract. This keeps GRPO group advantages
        # informative without granting correctness credit for prose alone.
        normalized = solution_str.lower()
        expected_status = str(ground_truth.get("status", "")).lower()
        expected_ids = {str(item) for item in ground_truth.get("citation_ids", [])}
        status_hint = float(bool(expected_status) and expected_status in normalized)
        citation_hint = (
            sum(item.lower() in normalized for item in expected_ids) / len(expected_ids)
            if expected_ids
            else float("citation_ids" in normalized or "[]" in normalized)
        )
        json_contract_hint = float("citation_ids" in normalized and "status" in normalized)
        violates_safety = bool(
            ground_truth.get("must_not_answer", False) and re.search(r"\banswered\b", normalized)
        )
        safety_hint = float(not violates_safety)
        correctness_hint = (
            0.25 * status_hint
            + 0.35 * citation_hint
            + 0.15 * json_contract_hint
            + 0.05 * safety_hint
        )
        return round(
            0.98 * correctness_hint + 0.02 * _trajectory_efficiency(solution_str),
            6,
        )
    expected_status = str(ground_truth.get("status", ""))
    status_score = float(predicted.get("status") == expected_status)
    expected_ids = {str(item) for item in ground_truth.get("citation_ids", [])}
    predicted_ids = {str(item) for item in predicted.get("citation_ids", [])}
    if not expected_ids and not predicted_ids:
        citation_score = 1.0
    elif not expected_ids or not predicted_ids:
        citation_score = 0.0
    else:
        true_positive = len(expected_ids & predicted_ids)
        precision = true_positive / len(predicted_ids)
        recall = true_positive / len(expected_ids)
        citation_score = 2 * precision * recall / (precision + recall) if true_positive else 0.0
    answer_safe = not (
        ground_truth.get("must_not_answer", False) and predicted.get("status") == "answered"
    )
    safety_score = float(answer_safe)
    correctness = 0.4 * status_score + 0.35 * citation_score + 0.25 * safety_score
    return round(
        0.98 * correctness + 0.02 * _trajectory_efficiency(solution_str),
        6,
    )
