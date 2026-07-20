"""Verify the local AutoDL export against its remote SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath

REMOTE_BASE = PurePosixPath("/root/autodl-tmp/evidenceagent-mm")
MODEL_MAPPINGS = {
    PurePosixPath(
        "/root/autodl-tmp/evidenceagent-mm/cache/huggingface/hub/"
        "models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    ): Path("models/Qwen--Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218"),
    PurePosixPath(
        "/root/autodl-tmp/evidenceagent-mm/cache/huggingface/hub/"
        "models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
    ): Path("models/BAAI--bge-m3/5617a9f61b028005a4858fdac845db406aefb181"),
    PurePosixPath(
        "/root/autodl-tmp/evidenceagent-mm/cache/huggingface/hub/"
        "models--Systran--faster-whisper-small/snapshots/"
        "536b0662742c02347bc0e980a01041f333bce120"
    ): Path("models/Systran--faster-whisper-small/536b0662742c02347bc0e980a01041f333bce120"),
    PurePosixPath("/root/.paddlex/official_models/PP-OCRv5_mobile_det"): Path(
        "models/PaddleOCR/PP-OCRv5_mobile_det"
    ),
    PurePosixPath("/root/.paddlex/official_models/PP-OCRv5_mobile_rec"): Path(
        "models/PaddleOCR/PP-OCRv5_mobile_rec"
    ),
}


def local_relative_path(remote_path: str) -> Path:
    """Map an audited remote path to its location in the local archive."""

    candidate = PurePosixPath(remote_path)
    for remote_prefix, local_prefix in MODEL_MAPPINGS.items():
        if candidate == remote_prefix or remote_prefix in candidate.parents:
            return local_prefix / Path(*candidate.relative_to(remote_prefix).parts)
    if candidate == REMOTE_BASE or REMOTE_BASE in candidate.parents:
        return Path(*candidate.relative_to(REMOTE_BASE).parts)
    raise ValueError(f"path is outside the audited export roots: {remote_path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify(archive_root: Path, manifest: Path) -> tuple[int, int]:
    checked = 0
    checked_bytes = 0
    failures: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, remote_path = line.split("  ", maxsplit=1)
        local_path = archive_root / local_relative_path(remote_path)
        if not local_path.is_file():
            failures.append(f"missing: {local_path}")
            continue
        actual = sha256(local_path)
        if actual != expected:
            failures.append(
                f"hash mismatch: {local_path}\n  expected {expected}\n  actual   {actual}"
            )
            continue
        checked += 1
        checked_bytes += local_path.stat().st_size
        print(f"OK {local_path.relative_to(archive_root)}")
    if failures:
        raise RuntimeError("archive verification failed:\n" + "\n".join(failures))
    return checked, checked_bytes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = args.manifest or (
        args.archive_root / "artifacts" / "export-2026-07-20" / "sha256-archive-final.txt"
    )
    checked, checked_bytes = verify(args.archive_root, manifest)
    print(f"VERIFIED files={checked} bytes={checked_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
