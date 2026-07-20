from __future__ import annotations

from evidenceagent_mm.retrieval import HashingEncoder, HybridRetriever
from evidenceagent_mm.store import EvidenceStore


def test_store_round_trip_and_parameterized_lookup(store: EvidenceStore) -> None:
    atom = store.get_atom("utt-01")
    assert atom is not None
    assert atom.speaker_id == "SPEAKER_00"
    assert store.get_atom("x' OR 1=1 --") is None


def test_hybrid_retrieval_returns_proposal_and_slide(store: EvidenceStore) -> None:
    retriever = HybridRetriever(store, HashingEncoder(64))
    hits = retriever.search("demo-session", "who proposed design B and which slide", top_k=3)
    ids = [hit.atom.evidence_id for hit in hits]
    assert "utt-01" in ids
    assert "ocr-01" in ids
    assert hits[0].score == 1.0
