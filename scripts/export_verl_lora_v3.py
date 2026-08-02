"""Export a compact PEFT adapter from a VERL/FSDP actor checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = torch.load(
        args.checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if not isinstance(state, dict):
        raise TypeError("VERL actor checkpoint must be a mapping")
    adapter = {
        key.replace(".default.weight", ".weight"): value.contiguous()
        for key, value in state.items()
        if "lora_A.default.weight" in key or "lora_B.default.weight" in key
    }
    if not adapter:
        raise ValueError("no LoRA tensors found in checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(adapter, args.output / "adapter_model.safetensors")
    target_modules = sorted({key.split(".lora_", 1)[0].rsplit(".", 1)[-1] for key in adapter})
    config = {
        "base_model_name_or_path": args.base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": args.alpha,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": args.rank,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
    }
    (args.output / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "source_checkpoint": str(args.checkpoint),
        "base_model": args.base_model,
        "tensor_count": len(adapter),
        "rank": args.rank,
        "alpha": args.alpha,
        "target_modules": target_modules,
        "claim_boundary": (
            "Exported from the verified step-50 VERL actor checkpoint. "
            "The run completed the actor update and checkpoint save; the final "
            "post-save FSDP-to-vLLM synchronization OOMed."
        ),
    }
    (args.output / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
