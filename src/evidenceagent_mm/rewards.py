"""Offline-verifiable rewards shared by GRPO training and evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from evidenceagent_mm.schema import EvidenceAtom, ResponseStatus
from evidenceagent_mm.training_schema import (
    AgentTrainingTarget,
    RewardWeights,
)

PARTIAL_REWARD_CAP = 0.75
TARGET_KEYS = {
    "status",
    "answer",
    "claims",
    "citation_ids",
    "missing_evidence",
    "clarifying_question",
    "confidence",
}


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    format: float
    status: float
    citation: float
    grounding: float
    abstention: float
    parse_error: str | None = None

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "total": self.total,
            "format": self.format,
            "status": self.status,
            "citation": self.citation,
            "grounding": self.grounding,
            "abstention": self.abstention,
            "parse_error": self.parse_error,
        }


def parse_completion_payload(completion: str) -> dict[str, Any]:
    text = completion.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("completion JSON must be an object")
    return payload


def parse_completion(completion: str) -> AgentTrainingTarget:
    return AgentTrainingTarget.model_validate(parse_completion_payload(completion))


def _f1(predicted: set[str], expected: set[str]) -> float:
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    true_positive = len(predicted & expected)
    precision = true_positive / len(predicted)
    recall = true_positive / len(expected)
    return 2 * precision * recall / (precision + recall) if true_positive else 0.0


def _grounding_score(prediction: AgentTrainingTarget, evidence: list[EvidenceAtom]) -> float:
    if prediction.status is not ResponseStatus.ANSWERED:
        return 1.0
    available = {atom.evidence_id for atom in evidence}
    cited = set(prediction.citation_ids)
    if not cited or not cited <= available:
        return 0.0
    claim_links = [
        set(claim.evidence_ids) <= cited and bool(set(claim.evidence_ids))
        for claim in prediction.claims
    ]
    if not claim_links:
        return 0.0
    # The schema proves claim-to-citation linkage. This additional term proves the
    # citations actually exist in the supplied context.
    return sum(claim_links) / len(claim_links)


def _abstention_score(prediction: AgentTrainingTarget, reference: AgentTrainingTarget) -> float:
    if reference.status is ResponseStatus.ANSWERED:
        return 1.0 if prediction.status is ResponseStatus.ANSWERED else 0.0
    if reference.status is ResponseStatus.ABSTAINED:
        if prediction.status is not ResponseStatus.ABSTAINED:
            return 0.0
        return _f1(set(prediction.missing_evidence), set(reference.missing_evidence))
    if prediction.status is not ResponseStatus.NEEDS_CLARIFICATION:
        return 0.0
    return 1.0 if prediction.clarifying_question else 0.0


def _partial_format_score(payload: dict[str, Any]) -> float:
    checks = {
        "status": isinstance(payload.get("status"), str),
        "answer": payload.get("answer") is None or isinstance(payload.get("answer"), str),
        "claims": isinstance(payload.get("claims"), list),
        "citation_ids": isinstance(payload.get("citation_ids"), list),
        "missing_evidence": isinstance(payload.get("missing_evidence"), list),
        "clarifying_question": (
            payload.get("clarifying_question") is None
            or isinstance(payload.get("clarifying_question"), str)
        ),
        "confidence": (
            isinstance(payload.get("confidence"), int | float)
            and not isinstance(payload.get("confidence"), bool)
            and 0 <= float(payload["confidence"]) <= 1
        ),
    }
    type_score = sum(checks.values()) / len(checks)
    exact_keys = float(set(payload) == TARGET_KEYS)
    return 0.8 * type_score + 0.2 * exact_keys


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str) and item}


def _partial_grounding_score(payload: dict[str, Any], evidence: list[EvidenceAtom]) -> float:
    if payload.get("status") != ResponseStatus.ANSWERED.value:
        return float(not _string_set(payload.get("citation_ids")) and not payload.get("claims"))
    available = {atom.evidence_id for atom in evidence}
    cited = _string_set(payload.get("citation_ids"))
    citation_validity = float(bool(cited) and cited <= available)
    claims = payload.get("claims")
    if not isinstance(claims, list) or not claims:
        return 0.5 * citation_validity
    claim_links: list[float] = []
    for claim in claims:
        if not isinstance(claim, dict):
            claim_links.append(0.0)
            continue
        claim_ids = _string_set(claim.get("evidence_ids"))
        linked = bool(claim_ids) and claim_ids <= cited and claim_ids <= available
        claim_links.append(float(linked))
    return 0.5 * citation_validity + 0.5 * statistics_mean(claim_links)


def statistics_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _partial_abstention_score(
    payload: dict[str, Any],
    reference: AgentTrainingTarget,
) -> float:
    status = payload.get("status")
    if reference.status is ResponseStatus.ANSWERED:
        return float(status == ResponseStatus.ANSWERED.value)
    if reference.status is ResponseStatus.ABSTAINED:
        if status != ResponseStatus.ABSTAINED.value:
            return 0.0
        return _f1(
            _string_set(payload.get("missing_evidence")),
            set(reference.missing_evidence),
        )
    return float(
        status == ResponseStatus.NEEDS_CLARIFICATION.value
        and isinstance(payload.get("clarifying_question"), str)
        and bool(payload["clarifying_question"].strip())
    )


def _score_partial_payload(
    payload: dict[str, Any],
    *,
    reference: AgentTrainingTarget,
    evidence: list[EvidenceAtom],
    weights: RewardWeights,
    error: Exception,
) -> RewardBreakdown:
    status_matches = payload.get("status") == reference.status.value
    citation = (
        _f1(_string_set(payload.get("citation_ids")), set(reference.citation_ids))
        if status_matches
        else 0.0
    )
    components = {
        "format": _partial_format_score(payload),
        "status": float(status_matches),
        "citation": citation,
        "grounding": (_partial_grounding_score(payload, evidence) if status_matches else 0.0),
        "abstention": _partial_abstention_score(payload, reference),
    }
    weighted = (
        weights.format * components["format"]
        + weights.status * components["status"]
        + weights.citation * components["citation"]
        + weights.grounding * components["grounding"]
        + weights.abstention * components["abstention"]
    ) / weights.total
    return RewardBreakdown(
        total=min(PARTIAL_REWARD_CAP, weighted),
        format=components["format"],
        status=components["status"],
        citation=components["citation"],
        grounding=components["grounding"],
        abstention=components["abstention"],
        parse_error=str(error),
    )


def score_completion(
    completion: str,
    *,
    reference: AgentTrainingTarget,
    evidence: list[EvidenceAtom],
    weights: RewardWeights | None = None,
) -> RewardBreakdown:
    active_weights = weights or RewardWeights()
    try:
        payload = parse_completion_payload(completion)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return RewardBreakdown(
            total=0.0,
            format=0.0,
            status=0.0,
            citation=0.0,
            grounding=0.0,
            abstention=0.0,
            parse_error=str(exc),
        )
    try:
        prediction = AgentTrainingTarget.model_validate(payload)
    except ValidationError as exc:
        return _score_partial_payload(
            payload,
            reference=reference,
            evidence=evidence,
            weights=active_weights,
            error=exc,
        )
    status_matches = prediction.status is reference.status
    components = {
        "format": 1.0,
        "status": float(status_matches),
        "citation": (
            _f1(set(prediction.citation_ids), set(reference.citation_ids))
            if status_matches
            else 0.0
        ),
        "grounding": _grounding_score(prediction, evidence) if status_matches else 0.0,
        "abstention": _abstention_score(prediction, reference),
    }
    weighted = (
        active_weights.format * components["format"]
        + active_weights.status * components["status"]
        + active_weights.citation * components["citation"]
        + active_weights.grounding * components["grounding"]
        + active_weights.abstention * components["abstention"]
    )
    return RewardBreakdown(
        total=weighted / active_weights.total,
        format=components["format"],
        status=components["status"],
        citation=components["citation"],
        grounding=components["grounding"],
        abstention=components["abstention"],
    )


def trl_verifiable_reward(
    completions: list[Any],
    reference_json: list[str],
    evidence_json: list[str],
    reward_weights_json: list[str] | None = None,
    **_: Any,
) -> list[float]:
    """TRL-compatible batched reward function with no model-based judge."""

    scores: list[float] = []
    for index, completion in enumerate(completions):
        if isinstance(completion, list):
            completion_text = str(completion[-1].get("content", ""))
        else:
            completion_text = str(completion)
        reference = AgentTrainingTarget.model_validate_json(reference_json[index])
        evidence_payload = json.loads(evidence_json[index])
        evidence = [EvidenceAtom.model_validate(item) for item in evidence_payload]
        weights = (
            RewardWeights.model_validate_json(reward_weights_json[index])
            if reward_weights_json is not None
            else RewardWeights()
        )
        scores.append(
            score_completion(
                completion_text,
                reference=reference,
                evidence=evidence,
                weights=weights,
            ).total
        )
    return scores
