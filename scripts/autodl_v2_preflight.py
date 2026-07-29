from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from evidenceagent_mm.training import load_training_config, validate_training_run


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_preflight(
    *,
    allow_cpu: bool,
    deepspeed_config: str | None = None,
    world_size: int = 1,
) -> dict[str, Any]:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    root = Path(__file__).resolve().parents[1]
    disk = shutil.disk_usage(root)
    config_reports = []
    for stage in ("sft", "dpo", "grpo"):
        config = load_training_config(root / "configs" / f"{stage}_4090.json")
        config = config.__class__.model_validate(
            {
                **config.model_dump(),
                "dataset_path": str(root / config.dataset_path),
                "deepspeed_config": deepspeed_config,
                "load_in_4bit": False if deepspeed_config else config.load_in_4bit,
                "launcher_world_size": world_size,
            }
        )
        config_reports.append(validate_training_run(config))
    gpu: dict[str, Any] = {"available": False}
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        gpu_lines = query.splitlines()
        name, memory_mib, driver = [value.strip() for value in gpu_lines[0].split(",")]
        gpu = {
            "available": True,
            "count": len(gpu_lines),
            "name": name,
            "memory_total_mib": int(memory_mib),
            "driver": driver,
        }
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError) as exc:
        if not allow_cpu:
            raise RuntimeError("nvidia-smi did not report a usable GPU") from exc
    hardware_ready = bool(gpu["available"] and gpu["memory_total_mib"] >= 20_000)
    if gpu["available"] and not hardware_ready and not allow_cpu:
        raise RuntimeError("at least 20 GiB VRAM is required by the supplied 4090 configs")
    required_packages = ["torch", "transformers", "trl", "peft", "datasets", "bitsandbytes"]
    if deepspeed_config:
        required_packages.append("deepspeed")
    packages = {name: _package_version(name) for name in required_packages}
    missing = [name for name, version in packages.items() if version is None]
    if missing and not allow_cpu:
        raise RuntimeError(f"training packages missing: {', '.join(missing)}")
    torch_cuda_available = False
    if packages["torch"] is not None:
        import torch

        torch_cuda_available = bool(torch.cuda.is_available())
    if not torch_cuda_available and not allow_cpu:
        raise RuntimeError("Torch is installed but torch.cuda.is_available() is false")
    gpu["torch_cuda_available"] = torch_cuda_available
    gpu_ready = hardware_ready and torch_cuda_available
    if gpu.get("count", 0) < world_size and not allow_cpu:
        raise RuntimeError(
            f"requested world_size={world_size}, but nvidia-smi reported "
            f"{gpu.get('count', 0)} GPU(s)"
        )
    report = {
        "status": "ready" if gpu_ready and not missing else "local_validation_only",
        "python": sys.version,
        "platform": platform.platform(),
        "disk_free_gib": round(disk.free / 1024**3, 2),
        "gpu": gpu,
        "packages": packages,
        "configs": config_reports,
        "requested_world_size": world_size,
        "deepspeed_config": deepspeed_config,
        "distributed_evidence": (
            "single_process_compatibility_only"
            if world_size == 1
            else "configuration_only_until_real_multi_gpu_run_completes"
        ),
    }
    if disk.free < 35 * 1024**3 and not allow_cpu:
        raise RuntimeError("at least 35 GiB free disk is required for models and checkpoints")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output", default="outputs/preflight_v2.json")
    parser.add_argument("--deepspeed-config")
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()
    report = collect_preflight(
        allow_cpu=args.allow_cpu,
        deepspeed_config=args.deepspeed_config,
        world_size=args.world_size,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
