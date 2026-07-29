"""Deterministic construction of SFT, DPO, and GRPO examples."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evidenceagent_mm.pipeline import load_fixture
from evidenceagent_mm.schema import EvidenceAtom, ResponseStatus
from evidenceagent_mm.training_schema import (
    AgentTrainingTarget,
    ChatMessage,
    DPOExample,
    GRPOExample,
    SFTExample,
    TrainingClaim,
    TrainingMetadata,
    stable_example_id,
    write_jsonl,
)

SYSTEM_PROMPT = """You are EvidenceAgent-MM. Use only the supplied evidence.
Return exactly one JSON object, without Markdown or additional keys, using this schema:
{
  "status": "answered" | "needs_clarification" | "abstained",
  "answer": string | null,
  "claims": [{"text": string, "evidence_ids": [string]}],
  "citation_ids": [string],
  "missing_evidence": [string],
  "clarifying_question": string | null,
  "confidence": number from 0 to 1
}
Every claims item must use exactly the keys text and evidence_ids. Cite exact supplied
evidence IDs. For answered, answer, claims, and citation_ids must be non-empty. For
needs_clarification, set answer=null, claims=[], citation_ids=[], and ask one targeted
question. For abstained, set answer=null, claims=[], citation_ids=[], and name the
missing evidence. Never use status values such as complete, success, or refused."""


class BenchmarkQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    session_id: str
    question: str
    expected_status: ResponseStatus
    gold_evidence_ids: list[str] = Field(default_factory=list)


def evidence_context(atoms: list[EvidenceAtom]) -> str:
    """Serialize provenance and acoustic attributes without dropping modalities."""

    rows: list[str] = []
    for atom in sorted(atoms, key=lambda item: (item.start_ms, item.evidence_id)):
        payload = {
            "evidence_id": atom.evidence_id,
            "modality": atom.modality.value,
            "time_ms": [atom.start_ms, atom.end_ms],
            "speaker_id": atom.speaker_id,
            "page_no": atom.page_no,
            "text": atom.text,
            "confidence": atom.confidence,
            "source_uri": atom.source_uri,
            "acoustic": atom.attributes.get("acoustic", {}),
            "asr": atom.attributes.get("asr", {}),
        }
        rows.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return "\n".join(rows)


def prompt_messages(question: str, atoms: list[EvidenceAtom]) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=f"Question:\n{question}\n\nEvidence JSONL:\n{evidence_context(atoms)}",
        ),
    ]


def build_target(question: BenchmarkQuestion, atoms: list[EvidenceAtom]) -> AgentTrainingTarget:
    by_id = {atom.evidence_id: atom for atom in atoms}
    if question.expected_status is ResponseStatus.ANSWERED:
        selected = [by_id[evidence_id] for evidence_id in question.gold_evidence_ids]
        facts = " ".join(atom.text.strip() for atom in selected if atom.text.strip())
        answer = facts or "The cited evidence supports the answer."
        return AgentTrainingTarget(
            status=ResponseStatus.ANSWERED,
            answer=answer,
            claims=[
                TrainingClaim(
                    text=answer,
                    evidence_ids=list(question.gold_evidence_ids),
                )
            ],
            citation_ids=list(question.gold_evidence_ids),
            confidence=min((atom.confidence for atom in selected), default=0.8),
        )
    if question.expected_status is ResponseStatus.NEEDS_CLARIFICATION:
        return AgentTrainingTarget(
            status=ResponseStatus.NEEDS_CLARIFICATION,
            clarifying_question=("Which specific speaker, proposal, or time range do you mean?"),
            missing_evidence=["referent_identity"],
            confidence=0.45,
        )
    return AgentTrainingTarget(
        status=ResponseStatus.ABSTAINED,
        missing_evidence=["relevant_evidence"],
        confidence=0.1,
    )


def _rejected_target(
    target: AgentTrainingTarget, atoms: list[EvidenceAtom]
) -> tuple[AgentTrainingTarget, str]:
    if target.status is ResponseStatus.ANSWERED:
        wrong_ids = [
            atom.evidence_id for atom in atoms if atom.evidence_id not in target.citation_ids
        ]
        if wrong_ids:
            payload = target.model_dump(mode="json")
            payload["citation_ids"] = [wrong_ids[0]]
            payload["claims"][0]["evidence_ids"] = [wrong_ids[0]]
            payload["confidence"] = 0.95
            return AgentTrainingTarget.model_validate(payload), "citation_correctness"
        return (
            AgentTrainingTarget(
                status=ResponseStatus.ABSTAINED,
                missing_evidence=["relevant_evidence"],
                confidence=0.2,
            ),
            "status_correctness",
        )
    available = [atom for atom in atoms if atom.text]
    if available:
        atom = available[0]
        invented = f"Unsupported answer inferred from {atom.text}"
        return (
            AgentTrainingTarget(
                status=ResponseStatus.ANSWERED,
                answer=invented,
                claims=[TrainingClaim(text=invented, evidence_ids=[atom.evidence_id])],
                citation_ids=[atom.evidence_id],
                confidence=0.95,
            ),
            "safe_abstention"
            if target.status is ResponseStatus.ABSTAINED
            else "targeted_clarification",
        )
    fallback = deepcopy(target.model_dump(mode="json"))
    fallback["confidence"] = 1.0 - target.confidence
    return AgentTrainingTarget.model_validate(fallback), "status_correctness"


def build_examples(
    question: BenchmarkQuestion,
    atoms: list[EvidenceAtom],
    *,
    source: str,
    source_license: str,
    split: str,
    seed: int,
) -> tuple[SFTExample, DPOExample, GRPOExample]:
    target = build_target(question, atoms)
    messages = prompt_messages(question.question, atoms)
    acoustic_conditions = summarize_acoustic_conditions(atoms)
    metadata = TrainingMetadata(
        source=source,
        split=split,
        license=source_license,
        seed=seed,
        acoustic_conditions=acoustic_conditions,
    )
    root_id = stable_example_id(question.question_id, question.session_id)
    rejected, reason = _rejected_target(target, atoms)
    sft = SFTExample(
        example_id=f"{root_id}-sft",
        session_id=question.session_id,
        messages=messages,
        evidence=atoms,
        target=target,
        metadata=metadata,
    )
    dpo = DPOExample(
        example_id=f"{root_id}-dpo",
        session_id=question.session_id,
        messages=messages,
        evidence=atoms,
        chosen=target,
        rejected=rejected,
        preference_reason=reason,
        metadata=metadata,
    )
    grpo = GRPOExample(
        example_id=f"{root_id}-grpo",
        session_id=question.session_id,
        messages=messages,
        evidence=atoms,
        reference=target,
        metadata=metadata,
    )
    return sft, dpo, grpo


def summarize_acoustic_conditions(atoms: list[EvidenceAtom]) -> dict[str, Any]:
    acoustic = [
        atom.attributes["acoustic"]
        for atom in atoms
        if isinstance(atom.attributes.get("acoustic"), dict)
    ]
    if not acoustic:
        return {"available": False}
    snr_values = [
        float(item["snr_db"]) for item in acoustic if isinstance(item.get("snr_db"), int | float)
    ]
    overlap_values = [
        float(item["overlap_probability"])
        for item in acoustic
        if isinstance(item.get("overlap_probability"), int | float)
    ]
    return {
        "available": True,
        "mean_snr_db": sum(snr_values) / len(snr_values) if snr_values else None,
        "max_overlap_probability": max(overlap_values) if overlap_values else None,
        "noise_types": sorted(
            {str(item["noise_type"]) for item in acoustic if item.get("noise_type")}
        ),
    }


def _split_for_session(session_id: str) -> str:
    suffix = session_id.rsplit("-", 1)[-1]
    bucket = (
        int(suffix) % 10
        if suffix.isdigit()
        else (int(stable_example_id(session_id).split("-")[1][:4], 16) % 10)
    )
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


def build_benchmark_training_data(
    benchmark_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 20260728,
) -> dict[str, Any]:
    benchmark_path = Path(benchmark_dir)
    fixture_paths = sorted((benchmark_path / "fixtures").glob("*.json"))
    fixtures = {fixture.session_id: fixture for fixture in map(load_fixture, fixture_paths)}
    questions = [
        BenchmarkQuestion.model_validate_json(line)
        for line in (benchmark_path / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stage_records: dict[str, list[Any]] = {"sft": [], "dpo": [], "grpo": []}
    split_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    for question in questions:
        fixture = fixtures[question.session_id]
        split = _split_for_session(question.session_id)
        split_counts[split] += 1
        built_examples = build_examples(
            question,
            fixture.atoms,
            source=f"{benchmark_path.name}:{question.question_id}",
            source_license=fixture.source_license,
            split=split,
            seed=seed,
        )
        for stage, record in zip(("sft", "dpo", "grpo"), built_examples, strict=True):
            stage_records[stage].append(record)
    destination = Path(output_dir)
    for stage, stage_examples in stage_records.items():
        write_jsonl(destination / f"{stage}.jsonl", stage_examples)
    manifest = {
        "schema_version": "2.0",
        "source": benchmark_path.as_posix(),
        "seed": seed,
        "examples_per_stage": len(questions),
        "split_counts_per_stage": split_counts,
        "files": {stage: f"{stage}.jsonl" for stage in stage_records},
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
