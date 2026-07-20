"""Verify the license-free VAD fallback and report its limits explicitly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidenceagent_mm.perception import EnergyTurnDetector


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return overlap / union if union else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav")
    parser.add_argument("manifest")
    parser.add_argument("--output", default="results/gpu/diarization_fallback_smoke.json")
    args = parser.parse_args()
    gold = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["gold_turns"]
    atoms = EnergyTurnDetector().detect(args.wav, "synthetic-diarization-smoke")
    pair_count = min(len(atoms), len(gold))
    ious = [
        interval_iou(
            (atoms[index].start_ms / 1_000, atoms[index].end_ms / 1_000),
            (gold[index]["start_seconds"], gold[index]["end_seconds"]),
        )
        for index in range(pair_count)
    ]
    result = {
        "backend": "energy-turn-detector",
        "is_full_speaker_diarization": False,
        "limitation": "detects speech turns; sequential IDs are not reusable speaker identities",
        "gold_turns": len(gold),
        "predicted_turns": len(atoms),
        "mean_temporal_iou": sum(ious) / len(ious) if ious else 0.0,
        "atoms": [atom.model_dump(mode="json") for atom in atoms],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
