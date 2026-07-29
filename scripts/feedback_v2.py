from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidenceagent_mm.flywheel import FeedbackStore, HardCaseFeedback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--store",
        default="data/interim/eamm_v2_hard_cases.jsonl",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add")
    add.add_argument("feedback_json")
    export = commands.add_parser("export")
    export.add_argument("--output", default="data/processed/eamm_v2_feedback_training")
    args = parser.parse_args()
    store = FeedbackStore(args.store)
    if args.command == "add":
        feedback = HardCaseFeedback.model_validate_json(
            Path(args.feedback_json).read_text(encoding="utf-8")
        )
        result = {
            "feedback_id": feedback.feedback_id,
            "inserted": store.append(feedback),
        }
    else:
        result = store.export_training_data(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
