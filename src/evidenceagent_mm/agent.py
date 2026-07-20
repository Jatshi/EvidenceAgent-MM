"""Bounded evidence agent with explicit answer, clarification, and abstention states."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Protocol

from evidenceagent_mm.retrieval import HybridRetriever
from evidenceagent_mm.schema import (
    AgentResponse,
    Citation,
    Claim,
    ConfidenceBreakdown,
    EvidenceAtom,
    Modality,
    ResponseStatus,
    RetrievalHit,
    ToolTrace,
)


@dataclass(frozen=True)
class GateConfig:
    min_retrieval_score: float = 0.35
    min_claim_support: float = 0.55
    ambiguity_margin: float = 0.05
    require_cross_modal_for_visual_questions: bool = True


class AnswerGenerator(Protocol):
    def generate(self, question: str, evidence: list[EvidenceAtom]) -> str: ...


class EvidenceAgent:
    """A deterministic, auditable baseline before adding a generative planner."""

    def __init__(
        self,
        retriever: HybridRetriever,
        config: GateConfig | None = None,
        generator: AnswerGenerator | None = None,
    ) -> None:
        self.retriever = retriever
        self.config = config or GateConfig()
        self.generator = generator

    def answer(self, session_id: str, question: str, *, top_k: int = 8) -> AgentResponse:
        trace_id = uuid.uuid4().hex
        started = time.perf_counter()
        hits = self.retriever.search(session_id, question, top_k=top_k, graph_hops=1)
        retrieve_ms = (time.perf_counter() - started) * 1_000
        trace = [
            ToolTrace(
                tool="hybrid_retrieve",
                input_summary=f"session={session_id}, top_k={top_k}",
                output_ids=[hit.atom.evidence_id for hit in hits],
                elapsed_ms=retrieve_ms,
            )
        ]

        if self._is_ambiguous(question, hits):
            return AgentResponse(
                status=ResponseStatus.NEEDS_CLARIFICATION,
                confidence=self._confidence(hits, claim_support=0.0, answerable=0.45),
                missing_evidence=["referent_identity"],
                clarifying_question="你指的是哪位说话人或哪个具体方案？请提供姓名、speaker ID 或时间范围。",
                trace=trace,
                trace_id=trace_id,
            )

        required = self._required_modalities(question)
        available = {hit.atom.modality for hit in hits if hit.score >= 0.25}
        missing = [
            modality.value for modality in sorted(required - available, key=lambda item: item.value)
        ]
        best_score = hits[0].score if hits else 0.0
        if not hits or best_score < self.config.min_retrieval_score or missing:
            reasons = missing or ["relevant_evidence"]
            return AgentResponse(
                status=ResponseStatus.ABSTAINED,
                confidence=self._confidence(hits, claim_support=0.0, answerable=0.1),
                missing_evidence=reasons,
                trace=trace,
                trace_id=trace_id,
            )

        selected = self._select_support(hits, required)
        support = self._support_score(selected, required)
        if support < self.config.min_claim_support:
            return AgentResponse(
                status=ResponseStatus.ABSTAINED,
                confidence=self._confidence(hits, claim_support=support, answerable=0.3),
                missing_evidence=["independent_claim_support"],
                trace=trace,
                trace_id=trace_id,
            )

        answer_text = (
            self.generator.generate(question, [hit.atom for hit in selected])
            if self.generator
            else self._render_answer(selected)
        )
        evidence_ids = [hit.atom.evidence_id for hit in selected]
        citations = [self._citation(hit.atom) for hit in selected]
        claim = Claim(
            claim_id="clm_01",
            text=answer_text,
            evidence_ids=evidence_ids,
            support_score=support,
        )
        trace.append(
            ToolTrace(
                tool="verify_claim_support",
                input_summary=f"claim=clm_01, evidence={len(selected)}",
                output_ids=evidence_ids,
                elapsed_ms=0.0,
            )
        )
        return AgentResponse(
            status=ResponseStatus.ANSWERED,
            answer=answer_text,
            claims=[claim],
            citations=citations,
            confidence=self._confidence(hits, claim_support=support, answerable=0.9),
            trace=trace,
            trace_id=trace_id,
        )

    @staticmethod
    def _is_ambiguous(question: str, hits: list[RetrievalHit]) -> bool:
        ambiguous_markers = ("他", "她", "那个方案", "这位老师", "the teacher", "that proposal")
        if not any(marker in question.lower() for marker in ambiguous_markers):
            return False
        speakers = {hit.atom.speaker_id for hit in hits[:5] if hit.atom.speaker_id}
        return len(speakers) > 1

    def _required_modalities(self, question: str) -> set[Modality]:
        lowered = question.lower()
        required: set[Modality] = set()
        if any(token in lowered for token in ("谁", "说", "提出", "who", "said", "speaker")):
            required.add(Modality.TRANSCRIPT)
        if self.config.require_cross_modal_for_visual_questions and any(
            token in lowered for token in ("屏幕", "哪一页", "幻灯片", "slide", "screen", "page")
        ):
            required.add(Modality.TRANSCRIPT)
            required.add(Modality.OCR)
        return required

    @staticmethod
    def _select_support(hits: list[RetrievalHit], required: set[Modality]) -> list[RetrievalHit]:
        if not required:
            return hits[:2]
        selected: list[RetrievalHit] = []
        for modality in sorted(required, key=lambda item: item.value):
            candidate = next((hit for hit in hits if hit.atom.modality is modality), None)
            if candidate and candidate not in selected:
                selected.append(candidate)
        return selected

    @staticmethod
    def _support_score(selected: list[RetrievalHit], required: set[Modality]) -> float:
        if not selected:
            return 0.0
        modality_coverage = len({hit.atom.modality for hit in selected} & required) / max(
            len(required), 1
        )
        evidence_quality = sum(hit.score * hit.atom.confidence for hit in selected) / len(selected)
        return max(0.0, min(1.0, 0.6 * evidence_quality + 0.4 * modality_coverage))

    @staticmethod
    def _render_answer(selected: list[RetrievalHit]) -> str:
        ordered = sorted(selected, key=lambda hit: (hit.atom.start_ms, hit.atom.modality.value))
        fragments: list[str] = []
        for hit in ordered:
            atom = hit.atom
            prefix = (
                f"{atom.speaker_id} 在 {atom.start_ms / 1000:.1f}s"
                if atom.speaker_id
                else f"{atom.start_ms / 1000:.1f}s"
            )
            page = f"（第 {atom.page_no} 页）" if atom.page_no else ""
            fragments.append(f"{prefix}{page}：{atom.text.strip()}")
        return "；".join(fragments)

    @staticmethod
    def _citation(atom: EvidenceAtom) -> Citation:
        return Citation(
            evidence_id=atom.evidence_id,
            modality=atom.modality,
            start_ms=atom.start_ms,
            end_ms=atom.end_ms,
            source_uri=atom.source_uri,
            speaker_id=atom.speaker_id,
            page_no=atom.page_no,
            quote=atom.text[:300],
        )

    @staticmethod
    def _confidence(
        hits: list[RetrievalHit], *, claim_support: float, answerable: float
    ) -> ConfidenceBreakdown:
        retrieval = hits[0].score if hits else 0.0
        alignment = max((hit.atom.confidence for hit in hits[:3]), default=0.0)
        # Baseline score only. A learned calibrator must replace this for benchmark claims.
        overall = 0.25 * answerable + 0.25 * retrieval + 0.2 * alignment + 0.3 * claim_support
        return ConfidenceBreakdown(
            answerability=answerable,
            retrieval=retrieval,
            alignment=alignment,
            claim_support=claim_support,
            calibrated_overall=max(0.0, min(1.0, overall)),
        )
