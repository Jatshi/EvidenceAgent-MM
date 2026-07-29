from __future__ import annotations

import json

from evidenceagent_mm.schema import EvidenceAtom, Modality, ResponseStatus
from evidenceagent_mm.training_data import (
    BenchmarkQuestion,
    build_benchmark_training_data,
    build_examples,
    evidence_context,
)
from evidenceagent_mm.training_schema import (
    DPOExample,
    GRPOExample,
    SFTExample,
    read_jsonl,
)


def test_build_examples_is_deterministic_and_preserves_acoustics() -> None:
    atom = EvidenceAtom(
        evidence_id="s:asr:1",
        session_id="s",
        modality=Modality.TRANSCRIPT,
        start_ms=0,
        end_ms=1000,
        text="Alice proposed the robust retriever.",
        source_uri="media://meeting.wav#t=0,1",
        confidence=0.9,
        attributes={
            "asr": {"backend": "faster-whisper", "model": "small"},
            "acoustic": {
                "snr_db": 3.0,
                "overlap_probability": 0.4,
                "noise_type": "babble",
            },
        },
    )
    question = BenchmarkQuestion(
        question_id="q1",
        session_id="s",
        question="Who proposed the retriever?",
        expected_status=ResponseStatus.ANSWERED,
        gold_evidence_ids=[atom.evidence_id],
    )

    first = build_examples(
        question, [atom], source="unit", source_license="CC0-1.0", split="train", seed=7
    )
    second = build_examples(
        question, [atom], source="unit", source_license="CC0-1.0", split="train", seed=7
    )

    assert first == second
    assert first[0].target.citation_ids == ["s:asr:1"]
    assert first[1].chosen != first[1].rejected
    assert first[2].metadata.acoustic_conditions["mean_snr_db"] == 3.0
    assert '"snr_db": 3.0' in evidence_context([atom])


def test_build_full_bronze_contracts(tmp_path) -> None:
    output = tmp_path / "training"
    manifest = build_benchmark_training_data("benchmarks/eamm_bronze", output, seed=11)

    assert manifest["examples_per_stage"] == 120
    assert sum(manifest["split_counts_per_stage"].values()) == 120
    assert len(read_jsonl(output / "sft.jsonl", SFTExample)) == 120
    assert len(read_jsonl(output / "dpo.jsonl", DPOExample)) == 120
    assert len(read_jsonl(output / "grpo.jsonl", GRPOExample)) == 120
    assert (
        json.loads((output / "manifest.json").read_text(encoding="utf-8"))["schema_version"]
        == "2.0"
    )
