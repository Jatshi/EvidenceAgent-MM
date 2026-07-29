from __future__ import annotations

from pathlib import Path

import pytest

try:
    import tomllib
except ImportError:  # pragma: no cover - exercised on Python 3.10 CI
    import tomli as tomllib

from scripts.train_v2 import build_parser


@pytest.mark.parametrize("option", ["--local-rank=0", "--local_rank=0"])
def test_training_cli_accepts_distributed_launcher_local_rank(option: str) -> None:
    args = build_parser().parse_args(["--config", "configs/sft_4090.json", option])
    assert args.local_rank == 0


def test_training_cli_accepts_chained_adapter_input() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "configs/dpo_4090.json",
            "--model-name-or-path",
            "/tmp/sft-adapter",
        ]
    )
    assert args.model_name_or_path == "/tmp/sft-adapter"


def test_distributed_extra_includes_deepspeed_jit_builder() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["optional-dependencies"]["distributed"]
    assert any(dependency.startswith("ninja") for dependency in dependencies)
