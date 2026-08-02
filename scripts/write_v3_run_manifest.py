"""Write an auditable run manifest after a successful v3 stage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_revision(root: Path) -> str | None:
    """Return an explicit or discovered revision without requiring a .git directory."""
    explicit = os.environ.get("EAMM_SOURCE_GIT_SHA")
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def source_tree_sha256(root: Path) -> str:
    """Hash the portable source bundle when Git metadata is intentionally absent."""
    digest = hashlib.sha256()
    source_roots = {"compat", "configs", "scripts", "src", "tests"}
    suffixes = {".json", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in suffixes
        and (
            len(path.relative_to(root).parts) == 1
            or path.relative_to(root).parts[0] in source_roots
        )
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    git_sha = git_revision(root)
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    missing = [str(path) for path in args.artifact if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required artifacts missing: {missing}")
    report = {
        "schema_version": "eamm.run_manifest.v3",
        "stage": args.stage,
        "status": "completed",
        "git_sha": git_sha,
        "source_tree_sha256": source_tree_sha256(root),
        "python": sys.version,
        "platform": platform.platform(),
        "gpu": gpu.splitlines(),
        "packages": {
            name: version(name)
            for name in ("torch", "transformers", "verl", "sglang", "datasets", "pyarrow")
        },
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in args.artifact
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
