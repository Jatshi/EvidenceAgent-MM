from __future__ import annotations

from evidenceagent_mm.agent import EvidenceAgent
from evidenceagent_mm.retrieval import HashingEncoder, HybridRetriever
from evidenceagent_mm.schema import ResponseStatus
from evidenceagent_mm.store import EvidenceStore


def make_agent(store: EvidenceStore) -> EvidenceAgent:
    return EvidenceAgent(HybridRetriever(store, HashingEncoder(64)))


def test_agent_answers_with_cross_modal_citations(store: EvidenceStore) -> None:
    response = make_agent(store).answer(
        "demo-session", "Who proposed design B and which slide page was on screen?"
    )
    assert response.status is ResponseStatus.ANSWERED
    assert {citation.evidence_id for citation in response.citations} == {"utt-01", "ocr-01"}
    assert response.claims[0].support_score >= 0.55


def test_agent_asks_for_ambiguous_referent(store: EvidenceStore) -> None:
    response = make_agent(store).answer("demo-session", "他提出的方案怎么样？")
    assert response.status is ResponseStatus.NEEDS_CLARIFICATION
    assert response.clarifying_question


def test_agent_abstains_when_evidence_is_missing(store: EvidenceStore) -> None:
    response = make_agent(store).answer("demo-session", "What was the final legal decision?")
    assert response.status is ResponseStatus.ABSTAINED
    assert response.missing_evidence


def test_agent_uses_injected_generator_after_verification(store: EvidenceStore) -> None:
    class FakeGenerator:
        def generate(self, question, evidence):
            assert "design B" in question
            assert {atom.evidence_id for atom in evidence} == {"utt-01", "ocr-01"}
            return "SPEAKER_00 proposed design B [utt-01] on page 7 [ocr-01]."

    agent = EvidenceAgent(HybridRetriever(store, HashingEncoder(64)), generator=FakeGenerator())
    response = agent.answer(
        "demo-session", "Who proposed design B and which slide page was on screen?"
    )
    assert response.status is ResponseStatus.ANSWERED
    assert response.answer and "[utt-01]" in response.answer
