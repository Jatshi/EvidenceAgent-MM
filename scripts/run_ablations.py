"""CLI wrapper for the deterministic EAMM ablation suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidenceagent_mm.experiments import run_ablation_suite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--work-dir", default="results/ablations/db")
    parser.add_argument("--output", default="benchmarks/results/ablations.json")
    args = parser.parse_args()
    report = run_ablation_suite(args.dataset, args.work_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["configs"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
