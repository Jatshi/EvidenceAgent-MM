"""Run faster-whisper on the synthetic video and persist provenance/VRAM metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from evidenceagent_mm.evaluation import word_error_rate
from evidenceagent_mm.perception import FasterWhisperASR


def gpu_state() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("media")
    parser.add_argument("--model", default="small")
    parser.add_argument("--output", default="results/gpu/asr_smoke.json")
    parser.add_argument("--reference-manifest")
    args = parser.parse_args()
    before = gpu_state()
    started = time.perf_counter()
    adapter = FasterWhisperASR(args.model)
    atoms = adapter.transcribe(args.media, "synthetic-gpu-smoke")
    elapsed = time.perf_counter() - started
    result = {
        "backend": "faster-whisper",
        "model": args.model,
        "gpu_before": before,
        "gpu_after": gpu_state(),
        "elapsed_seconds": elapsed,
        "atoms": [atom.model_dump(mode="json") for atom in atoms],
    }
    if args.reference_manifest:
        manifest = json.loads(Path(args.reference_manifest).read_text(encoding="utf-8"))
        reference = " ".join(manifest["transcript"])
        hypothesis = " ".join(atom.text for atom in atoms)
        result["reference"] = reference
        result["hypothesis"] = hypothesis
        result["wer"] = word_error_rate(reference, hypothesis)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"atoms": len(atoms), "elapsed_seconds": elapsed, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
