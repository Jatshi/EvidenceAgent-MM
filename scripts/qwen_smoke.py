"""Load Qwen3-8B and answer one question using only supplied evidence."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import torch
import transformers

from evidenceagent_mm.generation import QwenEvidenceGenerator
from evidenceagent_mm.provenance import huggingface_cache_revision
from evidenceagent_mm.schema import EvidenceAtom, Modality


def _citation_compliance(answer: str, atoms: list[EvidenceAtom]) -> tuple[bool, bool]:
    normalized = re.sub(r"[\W_]", "", answer.lower())
    citations_ok = all(f"[{atom.evidence_id}]" in answer.lower() for atom in atoms)
    facts_ok = all(fact in normalized for fact in ("speaker00", "page1", "42"))
    return citations_ok, facts_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output", default="results/gpu/qwen_smoke.json")
    args = parser.parse_args()
    atoms = [
        EvidenceAtom(
            evidence_id="gpu:utt:01",
            session_id="gpu-smoke",
            modality=Modality.TRANSCRIPT,
            start_ms=1_000,
            end_ms=4_000,
            speaker_id="SPEAKER_00",
            text="I propose design B because it reduces retrieval latency to 42 ms.",
            source_uri="media://meeting.mp4#t=1,4",
            confidence=0.96,
        ),
        EvidenceAtom(
            evidence_id="gpu:ocr:01",
            session_id="gpu-smoke",
            modality=Modality.OCR,
            start_ms=0,
            end_ms=5_000,
            page_no=1,
            text="Design B / P95 latency: 42 ms",
            source_uri="image://slide-1.png",
            confidence=0.94,
        ),
    ]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    generator = QwenEvidenceGenerator(args.model)
    loaded = time.perf_counter()
    question = "Who proposed design B, on which page, and why?"
    initial_answer = generator.generate(question, atoms)
    citations_ok, facts_ok = _citation_compliance(initial_answer, atoms)
    retry_used = not (citations_ok and facts_ok)
    if retry_used:
        question += (
            " Your response is invalid unless it copies these exact tokens: SPEAKER_00, "
            "page 1, 42 ms, [gpu:utt:01], and [gpu:ocr:01]. Return one sentence."
        )
        answer = generator.generate(question, atoms)
        citations_ok, facts_ok = _citation_compliance(answer, atoms)
    else:
        answer = initial_answer
    finished = time.perf_counter()
    result = {
        "model": args.model,
        "model_revision": huggingface_cache_revision(args.model),
        "initial_answer": initial_answer,
        "answer": answer,
        "retry_used": retry_used,
        "contains_both_evidence_ids": citations_ok,
        "contains_required_facts": facts_ok,
        "model_load_seconds": loaded - started,
        "generation_seconds": finished - loaded,
        "elapsed_seconds": finished - started,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return int(not (result["contains_both_evidence_ids"] and result["contains_required_facts"]))


if __name__ == "__main__":
    raise SystemExit(main())
