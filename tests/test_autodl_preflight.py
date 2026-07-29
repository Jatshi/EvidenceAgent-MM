from __future__ import annotations

from scripts.autodl_v2_preflight import collect_preflight


def test_preflight_supports_dependency_light_local_validation() -> None:
    report = collect_preflight(allow_cpu=True)
    assert report["status"] in {"ready", "local_validation_only"}
    assert {item["stage"] for item in report["configs"]} == {"sft", "dpo", "grpo"}


def test_preflight_reports_deepspeed_configuration_not_runtime_evidence() -> None:
    report = collect_preflight(
        allow_cpu=True,
        deepspeed_config="configs/deepspeed/zero3_cpu_offload.json",
        world_size=2,
    )
    assert report["requested_world_size"] == 2
    assert report["deepspeed_config"].endswith("zero3_cpu_offload.json")
    assert report["distributed_evidence"] == (
        "configuration_only_until_real_multi_gpu_run_completes"
    )
    assert all(item["deepspeed"]["zero_stage"] == 3 for item in report["configs"])
