"""Load Qwen3-8B and answer one question using only supplied evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from evidenceagent_mm.generation import QwenEvidenceGenerator
from evidenceagent_mm.schema import EvidenceAtom, Modality


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
    started = time.perf_counter()
    generator = QwenEvidenceGenerator(args.model)
    answer = generator.generate("Who proposed design B, on which page, and why?", atoms)
    elapsed = time.perf_counter() - started
    result = {
        "model": args.model,
        "answer": answer,
        "contains_both_evidence_ids": all(atom.evidence_id in answer for atom in atoms),
        "elapsed_seconds": elapsed,
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
