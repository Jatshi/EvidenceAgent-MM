"""Typed contracts shared by ingestion, retrieval, verification, and the API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Modality(str, Enum):
    TRANSCRIPT = "transcript"
    SLIDE = "slide"
    OCR = "ocr"
    AUDIO = "audio"
    FRAME = "frame"


class ResponseStatus(str, Enum):
    ANSWERED = "answered"
    NEEDS_CLARIFICATION = "needs_clarification"
    ABSTAINED = "abstained"


class EvidenceAtom(BaseModel):
    """Smallest independently citable unit in a meeting."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")
    session_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")
    modality: Modality
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: str = Field(default="", max_length=20_000)
    source_uri: str = Field(min_length=1)
    speaker_id: str | None = None
    page_no: int | None = Field(default=None, ge=1)
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_interval_and_bbox(self) -> EvidenceAtom:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError("bbox must be normalized as 0 <= x1 < x2 <= 1")
        return self


class EvidenceEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    relation: Literal[
        "next",
        "same_speaker",
        "overlaps",
        "supports",
        "shown_during",
        "mentions",
    ]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class RetrievalHit(BaseModel):
    atom: EvidenceAtom
    score: float = Field(ge=0.0)
    lexical_rank: int | None = None
    dense_rank: int | None = None
    graph_distance: int = Field(default=0, ge=0)


class Claim(BaseModel):
    claim_id: str
    text: str
    evidence_ids: list[str] = Field(min_length=1)
    support_score: float = Field(ge=0.0, le=1.0)


class Citation(BaseModel):
    evidence_id: str
    modality: Modality
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    source_uri: str
    speaker_id: str | None = None
    page_no: int | None = None
    quote: str = ""


class ConfidenceBreakdown(BaseModel):
    answerability: float = Field(ge=0.0, le=1.0)
    retrieval: float = Field(ge=0.0, le=1.0)
    alignment: float = Field(ge=0.0, le=1.0)
    claim_support: float = Field(ge=0.0, le=1.0)
    calibrated_overall: float = Field(ge=0.0, le=1.0)


class ToolTrace(BaseModel):
    tool: str
    input_summary: str
    output_ids: list[str] = Field(default_factory=list)
    elapsed_ms: float = Field(ge=0.0)


class AgentResponse(BaseModel):
    status: ResponseStatus
    answer: str | None = None
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: ConfidenceBreakdown
    missing_evidence: list[str] = Field(default_factory=list)
    clarifying_question: str | None = None
    trace: list[ToolTrace] = Field(default_factory=list)
    trace_id: str

    @model_validator(mode="after")
    def enforce_state_contract(self) -> AgentResponse:
        if self.status is ResponseStatus.ANSWERED:
            if not self.answer or not self.claims or not self.citations:
                raise ValueError("answered responses require answer, claims, and citations")
            cited = {citation.evidence_id for citation in self.citations}
            if any(not set(claim.evidence_ids) <= cited for claim in self.claims):
                raise ValueError("every claim evidence_id must have a citation")
        elif self.status is ResponseStatus.NEEDS_CLARIFICATION:
            if not self.clarifying_question:
                raise ValueError("clarification responses require a question")
            if self.answer or self.claims:
                raise ValueError("clarification responses cannot contain an answer")
        elif self.status is ResponseStatus.ABSTAINED:
            if self.answer or self.claims:
                raise ValueError("abstentions cannot contain an answer")
            if not self.missing_evidence:
                raise ValueError("abstentions must explain missing evidence")
        return self


class QueryRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=2, max_length=2_000)
    top_k: int = Field(default=8, ge=1, le=30)


class FixtureDocument(BaseModel):
    session_id: str
    title: str
    duration_ms: int = Field(gt=0)
    source_license: str
    atoms: list[EvidenceAtom] = Field(min_length=1)
    edges: list[EvidenceEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_membership(self) -> FixtureDocument:
        if any(atom.session_id != self.session_id for atom in self.atoms):
            raise ValueError("all atoms must belong to fixture session_id")
        ids = {atom.evidence_id for atom in self.atoms}
        for edge in self.edges:
            if edge.source_id not in ids or edge.target_id not in ids:
                raise ValueError("edge endpoint missing from fixture atoms")
        return self
