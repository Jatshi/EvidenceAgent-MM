from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def validate_artifact_dir(path: Path) -> dict[str, object]:
    manifest_path = path / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("run manifest does not prove a completed training run")
    adapter_files = list(path.glob("adapter_model.*"))
    if not adapter_files:
        raise ValueError("no adapter_model artifact found")
    return {
        "artifact_dir": str(path),
        "stage": manifest.get("stage"),
        "run_name": manifest.get("run_name"),
        "files": sorted(item.name for item in path.iterdir() if item.is_file()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    summary = validate_artifact_dir(artifact_dir)
    summary["repo_id"] = args.repo_id
    summary["private"] = args.private
    if args.dry_run:
        summary["status"] = "dry_run_validated"
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("set HF_TOKEN in the environment; never pass it on the command line")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
    result = api.upload_folder(
        repo_id=args.repo_id,
        folder_path=artifact_dir,
        commit_message=f"Upload {summary['run_name']} adapter and provenance",
        ignore_patterns=["checkpoint-*/*", "optimizer.pt", "scheduler.pt"],
    )
    summary["status"] = "uploaded"
    summary["commit_url"] = str(result)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
