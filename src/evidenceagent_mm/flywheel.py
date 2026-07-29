"""Append-only, deduplicated hard-case feedback and training-data export."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidenceagent_mm.schema import EvidenceAtom
from evidenceagent_mm.training_data import prompt_messages
from evidenceagent_mm.training_schema import (
    AgentTrainingTarget,
    DPOExample,
    GRPOExample,
    SFTExample,
    TrainingMetadata,
    stable_example_id,
    write_jsonl,
)


class HardCaseReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    USER_CORRECTION = "user_correction"
    EVIDENCE_CONFLICT = "evidence_conflict"
    WRONG_CITATION = "wrong_citation"
    WRONG_STATUS = "wrong_status"
    ASR_OR_OCR_ERROR = "asr_or_ocr_error"


class HardCaseFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "2.0"
    feedback_id: str | None = None
    session_id: str
    question: str = Field(min_length=2)
    reason: HardCaseReason
    evidence: list[EvidenceAtom]
    observed: AgentTrainingTarget
    corrected: AgentTrainingTarget | None = None
    reviewer: str = "anonymous"
    source_license: str
    consent_for_training: bool = False
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def assign_stable_id(self) -> HardCaseFeedback:
        if self.feedback_id is None:
            fingerprint = json.dumps(
                {
                    "session_id": self.session_id,
                    "question": self.question,
                    "reason": self.reason,
                    "observed": self.observed.model_dump(mode="json"),
                    "corrected": (
                        self.corrected.model_dump(mode="json") if self.corrected else None
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            self.feedback_id = (
                "feedback-" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:20]
            )
        return self


class FeedbackStore:
    """JSONL store suited to one-process annotation and deterministic export."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read_all(self) -> list[HardCaseFeedback]:
        if not self.path.exists():
            return []
        records: list[HardCaseFeedback] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(HardCaseFeedback.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{self.path}:{line_number}: invalid hard-case feedback") from exc
        return records

    def append(self, feedback: HardCaseFeedback) -> bool:
        existing = {record.feedback_id for record in self.read_all()}
        if feedback.feedback_id in existing:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as destination:
            destination.write(
                json.dumps(
                    feedback.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        return True

    def export_training_data(self, output_dir: str | Path) -> dict[str, Any]:
        eligible = [
            record
            for record in self.read_all()
            if record.consent_for_training and record.corrected is not None
        ]
        sft_records: list[SFTExample] = []
        dpo_records: list[DPOExample] = []
        grpo_records: list[GRPOExample] = []
        for record in eligible:
            assert record.corrected is not None
            metadata = TrainingMetadata(
                source=f"feedback:{record.feedback_id}",
                split="train",
                license=record.source_license,
                acoustic_conditions={
                    "hard_case_reason": record.reason.value,
                    **record.metadata,
                },
            )
            messages = prompt_messages(record.question, record.evidence)
            root_id = stable_example_id(record.feedback_id or "", record.session_id)
            sft_records.append(
                SFTExample(
                    example_id=f"{root_id}-sft",
                    session_id=record.session_id,
                    messages=messages,
                    evidence=record.evidence,
                    target=record.corrected,
                    metadata=metadata,
                )
            )
            if record.corrected != record.observed:
                dpo_records.append(
                    DPOExample(
                        example_id=f"{root_id}-dpo",
                        session_id=record.session_id,
                        messages=messages,
                        evidence=record.evidence,
                        chosen=record.corrected,
                        rejected=record.observed,
                        preference_reason=(
                            "citation_correctness"
                            if record.reason is HardCaseReason.WRONG_CITATION
                            else "status_correctness"
                        ),
                        metadata=metadata,
                    )
                )
            grpo_records.append(
                GRPOExample(
                    example_id=f"{root_id}-grpo",
                    session_id=record.session_id,
                    messages=messages,
                    evidence=record.evidence,
                    reference=record.corrected,
                    metadata=metadata,
                )
            )
        destination = Path(output_dir)
        write_jsonl(destination / "sft.jsonl", sft_records)
        write_jsonl(destination / "dpo.jsonl", dpo_records)
        write_jsonl(destination / "grpo.jsonl", grpo_records)
        manifest = {
            "schema_version": "2.0",
            "feedback_total": len(self.read_all()),
            "eligible": len(eligible),
            "exported": {
                "sft": len(sft_records),
                "dpo": len(dpo_records),
                "grpo": len(grpo_records),
            },
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest
