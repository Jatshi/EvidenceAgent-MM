"""Versioned contracts for EvidenceAgent-MM post-training data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidenceagent_mm.schema import EvidenceAtom, ResponseStatus

SCHEMA_VERSION: Literal["2.0"] = "2.0"


class TrainingStage(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class TrainingClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class AgentTrainingTarget(BaseModel):
    """Compact JSON target used by all post-training stages."""

    model_config = ConfigDict(extra="forbid")

    status: ResponseStatus
    answer: str | None = None
    claims: list[TrainingClaim] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    clarifying_question: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_state(self) -> AgentTrainingTarget:
        cited = set(self.citation_ids)
        if self.status is ResponseStatus.ANSWERED:
            if not self.answer or not self.claims or not self.citation_ids:
                raise ValueError("answered targets require answer, claims, and citations")
            if any(not set(claim.evidence_ids) <= cited for claim in self.claims):
                raise ValueError("claim evidence must be present in citation_ids")
        elif self.status is ResponseStatus.NEEDS_CLARIFICATION:
            if not self.clarifying_question:
                raise ValueError("clarification targets require clarifying_question")
            if self.answer or self.claims or self.citation_ids:
                raise ValueError("clarification targets cannot claim an answer")
        elif self.status is ResponseStatus.ABSTAINED:
            if not self.missing_evidence:
                raise ValueError("abstained targets require missing_evidence")
            if self.answer or self.claims or self.citation_ids:
                raise ValueError("abstained targets cannot claim an answer")
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class TrainingMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str
    split: Literal["train", "validation", "test"] = "train"
    license: str
    seed: int = 20260728
    acoustic_conditions: dict[str, Any] = Field(default_factory=dict)


class SFTExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = SCHEMA_VERSION
    stage: Literal[TrainingStage.SFT] = TrainingStage.SFT
    example_id: str = Field(min_length=8)
    session_id: str
    messages: list[ChatMessage] = Field(min_length=2)
    evidence: list[EvidenceAtom]
    target: AgentTrainingTarget
    metadata: TrainingMetadata


class DPOExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = SCHEMA_VERSION
    stage: Literal[TrainingStage.DPO] = TrainingStage.DPO
    example_id: str = Field(min_length=8)
    session_id: str
    messages: list[ChatMessage] = Field(min_length=2)
    evidence: list[EvidenceAtom]
    chosen: AgentTrainingTarget
    rejected: AgentTrainingTarget
    preference_reason: Literal[
        "citation_correctness",
        "status_correctness",
        "grounding",
        "safe_abstention",
        "targeted_clarification",
    ]
    metadata: TrainingMetadata

    @model_validator(mode="after")
    def chosen_must_differ(self) -> DPOExample:
        if self.chosen == self.rejected:
            raise ValueError("chosen and rejected targets must differ")
        return self


class RewardWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: float = Field(default=0.10, ge=0.0)
    status: float = Field(default=0.25, ge=0.0)
    citation: float = Field(default=0.25, ge=0.0)
    grounding: float = Field(default=0.25, ge=0.0)
    abstention: float = Field(default=0.15, ge=0.0)

    @model_validator(mode="after")
    def require_positive_total(self) -> RewardWeights:
        if self.total <= 0:
            raise ValueError("at least one reward weight must be positive")
        return self

    @property
    def total(self) -> float:
        return self.format + self.status + self.citation + self.grounding + self.abstention


class GRPOExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = SCHEMA_VERSION
    stage: Literal[TrainingStage.GRPO] = TrainingStage.GRPO
    example_id: str = Field(min_length=8)
    session_id: str
    messages: list[ChatMessage] = Field(min_length=2)
    evidence: list[EvidenceAtom]
    reference: AgentTrainingTarget
    reward_weights: RewardWeights = Field(default_factory=RewardWeights)
    metadata: TrainingMetadata


TrainingExample = SFTExample | DPOExample | GRPOExample
ExampleT = TypeVar("ExampleT", SFTExample, DPOExample, GRPOExample)


def stable_example_id(*parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"eamm2-{digest}"


def write_jsonl(path: str | Path, examples: Sequence[TrainingExample]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(example.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for example in examples
    )
    destination.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def read_jsonl(path: str | Path, model: type[ExampleT]) -> list[ExampleT]:
    records: list[ExampleT] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(model.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid {model.__name__}: {exc}") from exc
    return records
