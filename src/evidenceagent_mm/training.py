"""Optional TRL training runtime with dependency-light dry-run validation."""

from __future__ import annotations

import inspect
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidenceagent_mm.rewards import trl_verifiable_reward
from evidenceagent_mm.training_schema import (
    DPOExample,
    GRPOExample,
    SFTExample,
    TrainingExample,
    TrainingStage,
    read_jsonl,
)


class TrainingRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    run_name: str
    stage: TrainingStage
    model_name_or_path: str
    dataset_path: str
    output_dir: str
    seed: int = 20260728
    split: Literal["train", "validation", "test"] = "train"
    max_sequence_length: int = Field(default=2048, ge=256)
    max_prompt_length: int = Field(default=1536, ge=128)
    max_completion_length: int = Field(default=512, ge=64)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    num_train_epochs: float = Field(default=1.0, gt=0)
    max_steps: int = -1
    smoke_max_steps: int = Field(default=1, ge=1)
    logging_steps: int = Field(default=1, ge=1)
    save_steps: int = Field(default=25, ge=1)
    warmup_ratio: float = Field(default=0.03, ge=0.0, lt=1.0)
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = True
    load_in_4bit: bool = True
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    lora_target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    dpo_beta: float = Field(default=0.1, gt=0)
    grpo_num_generations: int = Field(default=4, ge=2)
    report_to: list[str] = Field(default_factory=list)
    deepspeed_config: str | None = None
    launcher_world_size: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_lengths(self) -> TrainingRunConfig:
        if self.max_prompt_length + self.max_completion_length > self.max_sequence_length:
            raise ValueError(
                "max_prompt_length + max_completion_length cannot exceed max_sequence_length"
            )
        if self.deepspeed_config is not None and self.load_in_4bit:
            raise ValueError(
                "DeepSpeed runs require load_in_4bit=false; ZeRO partitioning of "
                "bitsandbytes-quantized weights is not an accepted compatibility claim"
            )
        return self


def load_training_config(path: str | Path) -> TrainingRunConfig:
    return TrainingRunConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def validate_deepspeed_config(path: str | Path) -> dict[str, Any]:
    """Validate supported ZeRO/offload settings without importing DeepSpeed."""

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"DeepSpeed config not found: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid DeepSpeed JSON: {source}") from exc
    zero = payload.get("zero_optimization")
    if not isinstance(zero, dict):
        raise ValueError("DeepSpeed config requires zero_optimization")
    stage = zero.get("stage")
    if stage not in (2, 3):
        raise ValueError("only DeepSpeed ZeRO stage 2 or 3 is supported")
    optimizer_offload = zero.get("offload_optimizer")
    if not isinstance(optimizer_offload, dict) or optimizer_offload.get("device") != "cpu":
        raise ValueError("DeepSpeed config must offload optimizer state to CPU")
    parameter_offload = zero.get("offload_param")
    if stage == 3 and (
        not isinstance(parameter_offload, dict) or parameter_offload.get("device") != "cpu"
    ):
        raise ValueError("ZeRO-3 config must offload parameters to CPU")
    return {
        "path": source.as_posix(),
        "zero_stage": stage,
        "optimizer_offload": "cpu",
        "parameter_offload": (
            parameter_offload.get("device") if isinstance(parameter_offload, dict) else "none"
        ),
    }


def _records(config: TrainingRunConfig) -> list[SFTExample | DPOExample | GRPOExample]:
    model_by_stage = {
        TrainingStage.SFT: SFTExample,
        TrainingStage.DPO: DPOExample,
        TrainingStage.GRPO: GRPOExample,
    }
    model = model_by_stage[config.stage]
    records = cast(list[TrainingExample], read_jsonl(config.dataset_path, model))
    selected = [record for record in records if record.metadata.split == config.split]
    if not selected:
        raise ValueError(
            f"{config.dataset_path} contains no {config.split!r} records for {config.stage.value}"
        )
    return selected


def _render_messages(messages: list[Any]) -> str:
    return "\n".join(f"<|{message.role}|>\n{message.content}" for message in messages)


def _render_chat_template(
    tokenizer: Any,
    messages: list[Any],
    *,
    assistant_content: str | None = None,
    add_generation_prompt: bool = False,
) -> str:
    """Render training text with the selected model's native chat template.

    Qwen3 enables a long reasoning preamble by default. The alignment targets in
    this project are constrained JSON, so explicitly disabling thinking prevents
    GRPO generations from consuming the entire completion budget before emitting
    the JSON object that the deterministic reward function can score.
    """

    conversation = [{"role": message.role, "content": message.content} for message in messages]
    if assistant_content is not None:
        conversation.append({"role": "assistant", "content": assistant_content})
    rendered = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("tokenizer chat template returned empty training text")
    return rendered


def _training_tokenizer(config: TrainingRunConfig) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=False,
    )


def _is_adapter_checkpoint(model_name_or_path: str) -> bool:
    return (Path(model_name_or_path).expanduser() / "adapter_config.json").is_file()


def _trainer_model(config: TrainingRunConfig) -> tuple[Any, Any | None]:
    """Return either a fresh base-model ID or a trainable chained PEFT adapter."""

    if not _is_adapter_checkpoint(config.model_name_or_path):
        return config.model_name_or_path, _peft_config(config)

    from peft import PeftConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    adapter_path = Path(config.model_name_or_path).expanduser().resolve()
    adapter_config = PeftConfig.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        adapter_config.base_model_name_or_path,
        **_model_init_kwargs(config),
    )
    if config.load_in_4bit:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=True,
    )
    return model, None


def validate_training_run(config: TrainingRunConfig) -> dict[str, Any]:
    records = _records(config)
    prompt_lengths = [len(_render_messages(record.messages)) for record in records]
    example_ids = [record.example_id for record in records]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("dataset contains duplicate example_id values")
    deepspeed = (
        validate_deepspeed_config(config.deepspeed_config)
        if config.deepspeed_config is not None
        else None
    )
    return {
        "schema_version": config.schema_version,
        "run_name": config.run_name,
        "stage": config.stage.value,
        "model_name_or_path": config.model_name_or_path,
        "dataset_path": config.dataset_path,
        "split": config.split,
        "records": len(records),
        "max_prompt_characters": max(prompt_lengths),
        "mean_prompt_characters": sum(prompt_lengths) / len(prompt_lengths),
        "load_in_4bit": config.load_in_4bit,
        "estimated_mode": ("deepspeed_zero_lora" if config.deepspeed_config else "single_gpu_lora"),
        "deepspeed": deepspeed,
        "launcher_world_size": config.launcher_world_size,
        "distributed_evidence": (
            "single_process_compatibility_only"
            if config.launcher_world_size == 1
            else "configured_multi_process_not_yet_runtime_verified"
        ),
        "status": "dry_run_validated",
    }


def _supported_kwargs(callable_object: Any, values: dict[str, Any]) -> dict[str, Any]:
    """Keep compatibility across adjacent TRL versions without hiding required fields."""

    parameters = inspect.signature(callable_object).parameters
    if any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        return values
    return {key: value for key, value in values.items() if key in parameters}


def _model_init_kwargs(config: TrainingRunConfig) -> dict[str, Any]:
    import torch
    from transformers import BitsAndBytesConfig

    values: dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if config.bf16 else torch.float16,
        "trust_remote_code": False,
    }
    if config.deepspeed_config is None:
        values["device_map"] = "auto"
    if config.load_in_4bit:
        values["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return values


def _peft_config(config: TrainingRunConfig) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def _common_args(config: TrainingRunConfig, *, smoke: bool) -> dict[str, Any]:
    max_steps = config.smoke_max_steps if smoke else config.max_steps
    return {
        "output_dir": config.output_dir,
        "run_name": config.run_name + ("-smoke" if smoke else ""),
        "seed": config.seed,
        "data_seed": config.seed,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": max_steps,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_total_limit": 2,
        "warmup_ratio": config.warmup_ratio,
        "bf16": config.bf16,
        "tf32": config.tf32,
        "gradient_checkpointing": config.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "paged_adamw_8bit" if config.load_in_4bit else "adamw_torch_fused",
        "report_to": config.report_to,
        "remove_unused_columns": False,
        "model_init_kwargs": _model_init_kwargs(config),
        "deepspeed": config.deepspeed_config,
    }


def _dataset(rows: list[dict[str, Any]]) -> Any:
    from datasets import Dataset

    return Dataset.from_list(rows)


def _train_sft(config: TrainingRunConfig, records: list[Any], *, smoke: bool) -> Any:
    from trl import SFTConfig, SFTTrainer

    tokenizer = _training_tokenizer(config)
    model, peft_config = _trainer_model(config)
    rows = [
        {
            "text": _render_chat_template(
                tokenizer,
                record.messages,
                assistant_content=record.target.canonical_json(),
            )
        }
        for record in records
    ]
    values = {
        **_common_args(config, smoke=smoke),
        "max_length": config.max_sequence_length,
        "dataset_text_field": "text",
        "packing": False,
    }
    if not isinstance(model, str):
        values.pop("model_init_kwargs", None)
    args = SFTConfig(**_supported_kwargs(SFTConfig, values))
    return SFTTrainer(
        model=model,
        args=args,
        train_dataset=_dataset(rows),
        peft_config=peft_config,
        processing_class=tokenizer,
    )


def _train_dpo(config: TrainingRunConfig, records: list[Any], *, smoke: bool) -> Any:
    from trl import DPOConfig, DPOTrainer

    tokenizer = _training_tokenizer(config)
    model, peft_config = _trainer_model(config)
    rows = [
        {
            "prompt": _render_chat_template(
                tokenizer,
                record.messages,
                add_generation_prompt=True,
            ),
            "chosen": record.chosen.canonical_json(),
            "rejected": record.rejected.canonical_json(),
        }
        for record in records
    ]
    values = {
        **_common_args(config, smoke=smoke),
        "max_length": config.max_sequence_length,
        "max_prompt_length": config.max_prompt_length,
        "beta": config.dpo_beta,
    }
    if not isinstance(model, str):
        values.pop("model_init_kwargs", None)
    args = DPOConfig(**_supported_kwargs(DPOConfig, values))
    return DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=_dataset(rows),
        peft_config=peft_config,
        processing_class=tokenizer,
    )


def _train_grpo(config: TrainingRunConfig, records: list[Any], *, smoke: bool) -> Any:
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = _training_tokenizer(config)
    model, peft_config = _trainer_model(config)
    rows = [
        {
            "prompt": _render_chat_template(
                tokenizer,
                record.messages,
                add_generation_prompt=True,
            ),
            "reference_json": record.reference.canonical_json(),
            "evidence_json": json.dumps(
                [atom.model_dump(mode="json") for atom in record.evidence],
                ensure_ascii=False,
            ),
            "reward_weights_json": record.reward_weights.model_dump_json(),
        }
        for record in records
    ]
    values = {
        **_common_args(config, smoke=smoke),
        "max_prompt_length": config.max_prompt_length,
        "max_completion_length": config.max_completion_length,
        "num_generations": config.grpo_num_generations,
    }
    if not isinstance(model, str):
        values.pop("model_init_kwargs", None)
    args = GRPOConfig(**_supported_kwargs(GRPOConfig, values))
    return GRPOTrainer(
        model=model,
        reward_funcs=[trl_verifiable_reward],
        args=args,
        train_dataset=_dataset(rows),
        peft_config=peft_config,
        processing_class=tokenizer,
    )


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_training(config: TrainingRunConfig, *, smoke: bool = False) -> dict[str, Any]:
    try:
        import torch
        import transformers
        import trl
        from transformers import set_seed
    except ImportError as exc:
        raise RuntimeError(
            "training dependencies are missing; install evidenceagent-mm[train]"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured 4090 training run")
    deepspeed_version: str | None = None
    if config.deepspeed_config is not None:
        try:
            import deepspeed
        except ImportError as exc:
            raise RuntimeError(
                "DeepSpeed config selected; install evidenceagent-mm[train,distributed]"
            ) from exc
        deepspeed_version = deepspeed.__version__
    runtime_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if runtime_world_size != config.launcher_world_size:
        raise RuntimeError(
            f"launcher WORLD_SIZE={runtime_world_size} does not match "
            f"config launcher_world_size={config.launcher_world_size}"
        )
    set_seed(config.seed)
    records = _records(config)
    trainer_builders = {
        TrainingStage.SFT: _train_sft,
        TrainingStage.DPO: _train_dpo,
        TrainingStage.GRPO: _train_grpo,
    }
    trainer = trainer_builders[config.stage](config, records, smoke=smoke)
    result = trainer.train()
    trainer.save_model(config.output_dir)
    metrics = dict(result.metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    manifest = {
        **validate_training_run(config),
        "status": "completed",
        "smoke": smoke,
        "metrics": metrics,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "deepspeed": deepspeed_version,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(0),
            "world_size": runtime_world_size,
        },
        "git_revision": _git_revision(),
    }
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest
