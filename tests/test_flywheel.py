from __future__ import annotations

from evidenceagent_mm.flywheel import FeedbackStore, HardCaseFeedback, HardCaseReason
from evidenceagent_mm.schema import EvidenceAtom, Modality, ResponseStatus
from evidenceagent_mm.training_schema import AgentTrainingTarget, TrainingClaim


def test_feedback_store_deduplicates_and_exports_consented_corrections(tmp_path) -> None:
    atom = EvidenceAtom(
        evidence_id="e1",
        session_id="s",
        modality=Modality.TRANSCRIPT,
        start_ms=0,
        end_ms=1000,
        text="Alice proposed design B.",
        source_uri="media://x.wav#t=0,1",
    )
    observed = AgentTrainingTarget(
        status=ResponseStatus.ABSTAINED,
        missing_evidence=["relevant_evidence"],
        confidence=0.1,
    )
    corrected = AgentTrainingTarget(
        status=ResponseStatus.ANSWERED,
        answer="Alice proposed design B.",
        claims=[TrainingClaim(text="Alice proposed design B.", evidence_ids=["e1"])],
        citation_ids=["e1"],
        confidence=0.9,
    )
    feedback = HardCaseFeedback(
        session_id="s",
        question="Who proposed design B?",
        reason=HardCaseReason.WRONG_STATUS,
        evidence=[atom],
        observed=observed,
        corrected=corrected,
        source_license="CC-BY-4.0",
        consent_for_training=True,
        created_at="2026-07-28T00:00:00Z",
    )
    store = FeedbackStore(tmp_path / "feedback.jsonl")

    assert store.append(feedback)
    assert not store.append(feedback)
    manifest = store.export_training_data(tmp_path / "export")

    assert manifest["feedback_total"] == 1
    assert manifest["exported"] == {"sft": 1, "dpo": 1, "grpo": 1}
    assert (tmp_path / "export" / "sft.jsonl").read_text(encoding="utf-8")


def test_feedback_without_consent_is_not_exported(tmp_path) -> None:
    target = AgentTrainingTarget(
        status=ResponseStatus.ABSTAINED,
        missing_evidence=["relevant_evidence"],
    )
    feedback = HardCaseFeedback(
        session_id="s",
        question="Unknown question?",
        reason=HardCaseReason.LOW_CONFIDENCE,
        evidence=[],
        observed=target,
        corrected=target,
        source_license="private",
        consent_for_training=False,
        created_at="2026-07-28T00:00:00Z",
    )
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    store.append(feedback)
    manifest = store.export_training_data(tmp_path / "export")
    assert manifest["eligible"] == 0
