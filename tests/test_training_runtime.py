from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evidenceagent_mm.training import (
    TrainingRunConfig,
    _is_adapter_checkpoint,
    _render_chat_template,
    validate_deepspeed_config,
    validate_training_run,
)


def test_training_configs_validate_after_dataset_build(tmp_path) -> None:
    from evidenceagent_mm.training_data import build_benchmark_training_data

    destination = tmp_path / "training"
    build_benchmark_training_data("benchmarks/eamm_bronze", destination)
    for stage in ("sft", "dpo", "grpo"):
        config = TrainingRunConfig.model_validate(
            {
                "run_name": f"test-{stage}",
                "stage": stage,
                "model_name_or_path": "local/test-model",
                "dataset_path": str(destination / f"{stage}.jsonl"),
                "output_dir": str(tmp_path / "outputs" / stage),
            }
        )
        result = validate_training_run(config)
        assert result["records"] > 0
        assert result["status"] == "dry_run_validated"


def test_training_config_rejects_impossible_length_budget() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        TrainingRunConfig.model_validate(
            {
                "run_name": "bad",
                "stage": "sft",
                "model_name_or_path": "model",
                "dataset_path": "data.jsonl",
                "output_dir": "output",
                "max_sequence_length": 512,
                "max_prompt_length": 400,
                "max_completion_length": 200,
            }
        )


def test_repository_configs_are_json_and_stage_specific() -> None:
    for stage in ("sft", "dpo", "grpo"):
        payload = json.loads((Path("configs") / f"{stage}_4090.json").read_text(encoding="utf-8"))
        assert payload["stage"] == stage


def test_deepspeed_zero_configs_are_valid_cpu_offload() -> None:
    zero2 = validate_deepspeed_config("configs/deepspeed/zero2_cpu_offload.json")
    zero3 = validate_deepspeed_config("configs/deepspeed/zero3_cpu_offload.json")
    assert zero2["zero_stage"] == 2
    assert zero2["optimizer_offload"] == "cpu"
    assert zero2["parameter_offload"] == "none"
    assert zero3["zero_stage"] == 3
    assert zero3["parameter_offload"] == "cpu"


def test_deepspeed_dry_run_is_not_multi_gpu_evidence(tmp_path) -> None:
    from evidenceagent_mm.training_data import build_benchmark_training_data

    destination = tmp_path / "training"
    build_benchmark_training_data("benchmarks/eamm_bronze", destination)
    config = TrainingRunConfig(
        run_name="zero2-one-process",
        stage="sft",
        model_name_or_path="local/model",
        dataset_path=str(destination / "sft.jsonl"),
        output_dir=str(tmp_path / "output"),
        load_in_4bit=False,
        deepspeed_config="configs/deepspeed/zero2_cpu_offload.json",
        launcher_world_size=1,
    )
    report = validate_training_run(config)
    assert report["deepspeed"]["zero_stage"] == 2
    assert report["launcher_world_size"] == 1
    assert report["distributed_evidence"] == "single_process_compatibility_only"


def test_deepspeed_rejects_unaccepted_4bit_combination() -> None:
    with pytest.raises(ValueError, match="load_in_4bit=false"):
        TrainingRunConfig(
            run_name="invalid-zero",
            stage="sft",
            model_name_or_path="local/model",
            dataset_path="dataset.jsonl",
            output_dir="output",
            load_in_4bit=True,
            deepspeed_config="configs/deepspeed/zero2_cpu_offload.json",
        )


def test_deepspeed_validator_rejects_nonzero_config(tmp_path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps(
            {
                "zero_optimization": {
                    "stage": 1,
                    "offload_optimizer": {"device": "cpu"},
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stage 2 or 3"):
        validate_deepspeed_config(invalid)


def test_deepspeed_path_is_passed_to_training_arguments(monkeypatch) -> None:
    import evidenceagent_mm.training as training

    config = TrainingRunConfig(
        run_name="zero2-args",
        stage="dpo",
        model_name_or_path="local/model",
        dataset_path="dataset.jsonl",
        output_dir="output",
        load_in_4bit=False,
        deepspeed_config="configs/deepspeed/zero2_cpu_offload.json",
    )
    monkeypatch.setattr(training, "_model_init_kwargs", lambda _: {})
    arguments = training._common_args(config, smoke=True)
    assert arguments["deepspeed"] == "configs/deepspeed/zero2_cpu_offload.json"
    assert arguments["max_steps"] == 1


def test_native_chat_template_disables_qwen_thinking() -> None:
    calls: list[dict[str, object]] = []

    class StubTokenizer:
        def apply_chat_template(self, conversation, **kwargs):
            calls.append({"conversation": conversation, **kwargs})
            return "rendered-chat"

    messages = [
        SimpleNamespace(role="system", content="Return JSON only."),
        SimpleNamespace(role="user", content="Question"),
    ]
    rendered = _render_chat_template(
        StubTokenizer(),
        messages,
        add_generation_prompt=True,
    )
    assert rendered == "rendered-chat"
    assert calls[0]["enable_thinking"] is False
    assert calls[0]["add_generation_prompt"] is True
    assert calls[0]["conversation"] == [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": "Question"},
    ]


def test_native_chat_template_includes_sft_target_as_assistant() -> None:
    class StubTokenizer:
        def apply_chat_template(self, conversation, **kwargs):
            assert kwargs["enable_thinking"] is False
            assert kwargs["add_generation_prompt"] is False
            return json.dumps(conversation)

    rendered = _render_chat_template(
        StubTokenizer(),
        [SimpleNamespace(role="user", content="Question")],
        assistant_content='{"status":"answered"}',
    )
    assert json.loads(rendered)[-1] == {
        "role": "assistant",
        "content": '{"status":"answered"}',
    }


def test_training_prompt_spells_out_machine_checkable_schema() -> None:
    from evidenceagent_mm.training_data import SYSTEM_PROMPT

    assert '"status": "answered" | "needs_clarification" | "abstained"' in SYSTEM_PROMPT
    assert '"claims": [{"text": string, "evidence_ids": [string]}]' in SYSTEM_PROMPT
    assert "without Markdown or additional keys" in SYSTEM_PROMPT
    assert "Never use status values such as complete" in SYSTEM_PROMPT


def test_adapter_checkpoint_detection_requires_adapter_config(tmp_path) -> None:
    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    assert not _is_adapter_checkpoint(str(checkpoint))
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert _is_adapter_checkpoint(str(checkpoint))
