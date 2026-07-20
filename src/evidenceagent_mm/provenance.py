"""Small helpers for recording reproducible model provenance."""

from __future__ import annotations

import os
from pathlib import Path


def huggingface_cache_revision(model_id: str, cache_home: str | Path | None = None) -> str | None:
    """Resolve a cached Hugging Face commit without requiring network access."""

    if cache_home is None:
        cache_home = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    model_cache = Path(cache_home) / "hub" / f"models--{model_id.replace('/', '--')}"
    main_ref = model_cache / "refs" / "main"
    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        return revision or None
    snapshots = sorted(path.name for path in (model_cache / "snapshots").glob("*") if path.is_dir())
    return snapshots[0] if len(snapshots) == 1 else None
