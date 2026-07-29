from __future__ import annotations

import argparse
import json

from evidenceagent_mm.training_data import build_benchmark_training_data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="benchmarks/eamm_bronze")
    parser.add_argument("--output", default="benchmarks/eamm_v2_training")
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    result = build_benchmark_training_data(
        args.benchmark,
        args.output,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
