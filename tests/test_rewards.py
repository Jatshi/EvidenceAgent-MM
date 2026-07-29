from __future__ import annotations

from evidenceagent_mm.rewards import (
    PARTIAL_REWARD_CAP,
    score_completion,
    trl_verifiable_reward,
)
from evidenceagent_mm.schema import EvidenceAtom, Modality, ResponseStatus
from evidenceagent_mm.training_schema import AgentTrainingTarget, TrainingClaim


def answered_fixture() -> tuple[AgentTrainingTarget, list[EvidenceAtom]]:
    atom = EvidenceAtom(
        evidence_id="e1",
        session_id="s",
        modality=Modality.TRANSCRIPT,
        start_ms=0,
        end_ms=1000,
        text="Alice proposed design B.",
        source_uri="media://x.wav#t=0,1",
    )
    target = AgentTrainingTarget(
        status=ResponseStatus.ANSWERED,
        answer="Alice proposed design B.",
        claims=[
            TrainingClaim(
                text="Alice proposed design B.",
                evidence_ids=["e1"],
            )
        ],
        citation_ids=["e1"],
        confidence=0.9,
    )
    return target, [atom]


def test_perfect_completion_receives_full_verifiable_reward() -> None:
    target, atoms = answered_fixture()
    result = score_completion(
        target.canonical_json(),
        reference=target,
        evidence=atoms,
    )
    assert result.total == 1.0
    assert result.parse_error is None


def test_invalid_or_hallucinated_completion_is_penalized() -> None:
    target, atoms = answered_fixture()
    assert score_completion("not-json", reference=target, evidence=atoms).total == 0.0
    hallucinated = target.model_copy(
        update={
            "citation_ids": ["missing"],
            "claims": [TrainingClaim(text="Alice proposed design B.", evidence_ids=["missing"])],
        }
    )
    result = score_completion(
        hallucinated.canonical_json(),
        reference=target,
        evidence=atoms,
    )
    assert result.citation == 0.0
    assert result.grounding == 0.0
    assert result.total < 1.0


def test_safe_abstention_reward_checks_missing_evidence() -> None:
    reference = AgentTrainingTarget(
        status=ResponseStatus.ABSTAINED,
        missing_evidence=["legal_approval"],
        confidence=0.1,
    )
    good = score_completion(
        reference.canonical_json(),
        reference=reference,
        evidence=[],
    )
    wrong = AgentTrainingTarget(
        status=ResponseStatus.ABSTAINED,
        missing_evidence=["audio"],
        confidence=0.1,
    )
    bad = score_completion(wrong.canonical_json(), reference=reference, evidence=[])
    assert good.total == 1.0
    assert bad.abstention == 0.0


def test_over_refusal_is_penalized_for_answerable_reference() -> None:
    target, atoms = answered_fixture()
    refusal = AgentTrainingTarget(
        status=ResponseStatus.ABSTAINED,
        missing_evidence=["relevant_evidence"],
        confidence=0.1,
    )
    result = score_completion(
        refusal.canonical_json(),
        reference=target,
        evidence=atoms,
    )
    assert result.status == 0.0
    assert result.citation == 0.0
    assert result.grounding == 0.0
    assert result.abstention == 0.0
    assert result.total == 0.1


def test_trl_batch_adapter_is_deterministic() -> None:
    target, atoms = answered_fixture()
    scores = trl_verifiable_reward(
        [target.canonical_json()],
        [target.canonical_json()],
        ["[" + atoms[0].model_dump_json() + "]"],
    )
    assert scores == [1.0]


def test_near_valid_json_gets_capped_shaped_reward() -> None:
    target, atoms = answered_fixture()
    completion = """{
      "status": "answered",
      "answer": "Alice proposed design B.",
      "claims": [
        {"text": "Alice proposed design B.", "evidence_ids": ["e1"]},
        {"text": "Unsupported second claim.", "evidence_ids": ["e2"]}
      ],
      "citation_ids": ["e1"],
      "missing_evidence": [],
      "clarifying_question": null,
      "confidence": 0.8
    }"""
    result = score_completion(completion, reference=target, evidence=atoms)
    assert 0.0 < result.total <= PARTIAL_REWARD_CAP
    assert result.format == 1.0
    assert result.status == 1.0
    assert result.citation == 1.0
    assert result.grounding == 0.75
    assert result.parse_error is not None


def test_partial_reward_penalizes_wrong_field_types() -> None:
    target, atoms = answered_fixture()
    completion = """{
      "status": "answered",
      "answer": "Alice proposed design B.",
      "claims": [],
      "citation_ids": ["e1"],
      "missing_evidence": "audio",
      "clarifying_question": null,
      "confidence": 0.8
    }"""
    result = score_completion(completion, reference=target, evidence=atoms)
    assert 0.0 < result.format < 1.0
    assert result.total <= PARTIAL_REWARD_CAP
