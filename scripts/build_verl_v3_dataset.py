"""Convert EvidenceAgent v3 JSONL cases into verl tool-agent Parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evidenceagent_mm.verl_dataset import build_verl_agent_record


def read_cases(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {
            "sample_id",
            "session_id",
            "question",
            "ground_truth",
            "split",
            "evidence_store",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(f"line {line_number} missing fields: {sorted(missing)}")
        rows.append(row)
    if not rows:
        raise ValueError("case file is empty")
    return rows


def build_records(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = [build_verl_agent_record(**case) for case in cases]
    if len({record["extra_info"]["sample_id"] for record in records}) != len(records):
        raise ValueError("sample_id values must be unique")
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = build_records(read_cases(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import Dataset
    except ImportError as error:
        raise RuntimeError("Install datasets and pyarrow to write verl Parquet") from error
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        subset = [row for row in records if row["extra_info"]["split"] == split]
        if not subset:
            continue
        output = args.output_dir / f"{split}.parquet"
        Dataset.from_list(subset).to_parquet(str(output))
        counts[split] = len(subset)
    manifest = {
        "schema_version": "eamm.verl_dataset.v3",
        "source": str(args.input.resolve()),
        "counts": counts,
        "agent_name": "tool_agent",
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
