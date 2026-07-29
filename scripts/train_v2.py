from __future__ import annotations

import argparse
import json

from evidenceagent_mm.training import (
    TrainingRunConfig,
    load_training_config,
    run_training,
    validate_training_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--deepspeed-config")
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=-1,
        help="Process-local rank injected by torchrun/DeepSpeed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_training_config(args.config)
    updates: dict[str, object] = {}
    if args.output_dir:
        updates["output_dir"] = args.output_dir
    if args.model_name_or_path:
        updates["model_name_or_path"] = args.model_name_or_path
    if args.deepspeed_config:
        updates["deepspeed_config"] = args.deepspeed_config
    if args.no_4bit:
        updates["load_in_4bit"] = False
    if args.world_size:
        updates["launcher_world_size"] = args.world_size
    if updates:
        config = config.model_copy(update=updates)
    config = TrainingRunConfig.model_validate(config.model_dump())
    result = (
        validate_training_run(config) if args.dry_run else run_training(config, smoke=args.smoke)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
