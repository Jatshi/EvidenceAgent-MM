"""Deterministic held-out evaluation for a trained EvidenceAgent-MM adapter."""

from __future__ import annotations

import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Literal

from evidenceagent_mm.rewards import score_completion
from evidenceagent_mm.training import _render_chat_template
from evidenceagent_mm.training_schema import GRPOExample, read_jsonl

REWARD_COMPONENTS = ("total", "format", "status", "citation", "grounding", "abstention")


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty prediction set")
    metrics = {
        name: statistics.mean(float(row["reward"][name]) for row in rows)
        for name in REWARD_COMPONENTS
    }
    latencies = [float(row["latency_seconds"]) for row in rows]
    metrics.update(
        {
            "samples": len(rows),
            "valid_json_rate": statistics.mean(
                float(row["reward"].get("parse_error") is None) for row in rows
            ),
            "latency_seconds_mean": statistics.mean(latencies),
            "latency_seconds_p50": _percentile(latencies, 0.50),
            "latency_seconds_p95": _percentile(latencies, 0.95),
        }
    )
    return metrics


def _validate_adapter(path: Path) -> dict[str, Any]:
    manifest_path = path / "run_manifest.json"
    adapter_config = path / "adapter_config.json"
    adapter_weights = list(path.glob("adapter_model.*"))
    if not manifest_path.is_file() or not adapter_config.is_file() or not adapter_weights:
        raise ValueError(f"{path} is not a complete PEFT adapter artifact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"{manifest_path} does not prove a completed run")
    return manifest


def evaluate_adapter(
    adapter_dir: str | Path,
    dataset_path: str | Path,
    *,
    split: Literal["validation", "test"] = "test",
    output_dir: str | Path,
    max_samples: int | None = None,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    try:
        import torch
        import transformers
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install EvidenceAgent-MM training dependencies") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for adapter evaluation")

    adapter = Path(adapter_dir).expanduser().resolve()
    training_manifest = _validate_adapter(adapter)
    records = [
        record for record in read_jsonl(dataset_path, GRPOExample) if record.metadata.split == split
    ]
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        records = records[:max_samples]
    if not records:
        raise ValueError(f"no {split!r} records found in {dataset_path}")

    peft_config = PeftConfig.from_pretrained(adapter)
    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    device = next(model.parameters()).device

    rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    for record in records:
        prompt = _render_chat_template(
            tokenizer,
            record.messages,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1536,
        ).to(device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        completion = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        reward = score_completion(
            completion,
            reference=record.reference,
            evidence=record.evidence,
            weights=record.reward_weights,
        )
        rows.append(
            {
                "example_id": record.example_id,
                "session_id": record.session_id,
                "split": split,
                "completion": completion,
                "reference": record.reference.model_dump(mode="json"),
                "reward": reward.as_dict(),
                "latency_seconds": elapsed,
                "prompt_tokens": int(inputs["input_ids"].shape[1]),
                "completion_tokens": int(output.shape[1] - inputs["input_ids"].shape[1]),
            }
        )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    predictions_path = destination / f"{split}_predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "schema_version": "eamm.adapter_eval.v2",
        "status": "completed",
        "adapter_dir": str(adapter),
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "split": split,
        "predictions": str(predictions_path),
        "metrics": summarize_prediction_rows(rows),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(0),
        },
        "training_run": {
            "run_name": training_manifest.get("run_name"),
            "stage": training_manifest.get("stage"),
            "model_name_or_path": training_manifest.get("model_name_or_path"),
            "git_revision": training_manifest.get("git_revision"),
        },
    }
    report_path = destination / f"{split}_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
