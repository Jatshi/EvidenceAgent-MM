from __future__ import annotations

import argparse
import json

from evidenceagent_mm.adapter_evaluation import evaluate_adapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument(
        "--dataset",
        default="benchmarks/eamm_v2_training/grpo.jsonl",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = evaluate_adapter(
        args.adapter_dir,
        args.dataset,
        split=args.split,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
