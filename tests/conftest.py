from __future__ import annotations

import pytest

from evidenceagent_mm.pipeline import ingest_fixture
from evidenceagent_mm.schema import EvidenceAtom, EvidenceEdge, FixtureDocument, Modality
from evidenceagent_mm.store import EvidenceStore


@pytest.fixture
def fixture_document() -> FixtureDocument:
    session_id = "demo-session"
    return FixtureDocument(
        session_id=session_id,
        title="Synthetic design review",
        duration_ms=60_000,
        source_license="CC0-1.0",
        atoms=[
            EvidenceAtom(
                evidence_id="utt-01",
                session_id=session_id,
                modality=Modality.TRANSCRIPT,
                start_ms=10_000,
                end_ms=14_000,
                speaker_id="SPEAKER_00",
                text="I propose switching to design B because it reduces latency.",
                source_uri="media://demo.mp4#t=10,14",
                confidence=0.96,
            ),
            EvidenceAtom(
                evidence_id="ocr-01",
                session_id=session_id,
                modality=Modality.OCR,
                start_ms=9_000,
                end_ms=20_000,
                page_no=7,
                text="Design B: latency 42 ms",
                source_uri="image://slide-07.png",
                confidence=0.93,
            ),
            EvidenceAtom(
                evidence_id="utt-02",
                session_id=session_id,
                modality=Modality.TRANSCRIPT,
                start_ms=20_000,
                end_ms=24_000,
                speaker_id="SPEAKER_01",
                text="The budget review should happen next week.",
                source_uri="media://demo.mp4#t=20,24",
                confidence=0.94,
            ),
        ],
        edges=[
            EvidenceEdge(
                source_id="utt-01", target_id="ocr-01", relation="shown_during", confidence=0.93
            )
        ],
    )


@pytest.fixture
def store(tmp_path, fixture_document: FixtureDocument) -> EvidenceStore:
    evidence_store = EvidenceStore(tmp_path / "evidence.db")
    ingest_fixture(evidence_store, fixture_document)
    yield evidence_store
    evidence_store.close()
