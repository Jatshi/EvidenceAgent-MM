from __future__ import annotations

import json

import pytest

from scripts.publish_v2_adapter import validate_artifact_dir


def test_publish_requires_completed_manifest_and_adapter(tmp_path) -> None:
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"status": "completed", "stage": "sft", "run_name": "test"}),
        encoding="utf-8",
    )
    (tmp_path / "adapter_model.safetensors").write_bytes(b"adapter")
    result = validate_artifact_dir(tmp_path)
    assert result["stage"] == "sft"
    assert "adapter_model.safetensors" in result["files"]


def test_publish_rejects_unproven_artifacts(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing"):
        validate_artifact_dir(tmp_path)
