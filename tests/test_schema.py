from __future__ import annotations

import pytest
from pydantic import ValidationError

from evidenceagent_mm.schema import (
    AgentResponse,
    ConfidenceBreakdown,
    EvidenceAtom,
    Modality,
    ResponseStatus,
)


def confidence() -> ConfidenceBreakdown:
    return ConfidenceBreakdown(
        answerability=0.5,
        retrieval=0.5,
        alignment=0.5,
        claim_support=0.5,
        calibrated_overall=0.5,
    )


def test_atom_rejects_invalid_interval() -> None:
    with pytest.raises(ValidationError, match="end_ms"):
        EvidenceAtom(
            evidence_id="x",
            session_id="s",
            modality=Modality.AUDIO,
            start_ms=100,
            end_ms=100,
            source_uri="media://x",
        )


def test_answered_response_requires_citations() -> None:
    with pytest.raises(ValidationError, match="citations"):
        AgentResponse(
            status=ResponseStatus.ANSWERED,
            answer="unsupported",
            confidence=confidence(),
            trace_id="trace",
        )


def test_abstention_requires_missing_evidence() -> None:
    with pytest.raises(ValidationError, match="missing evidence"):
        AgentResponse(
            status=ResponseStatus.ABSTAINED,
            confidence=confidence(),
            trace_id="trace",
        )
