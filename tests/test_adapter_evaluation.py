from __future__ import annotations

import pytest

from evidenceagent_mm.adapter_evaluation import summarize_prediction_rows
from scripts.evaluate_v2_adapter import build_parser


def test_prediction_summary_aggregates_rewards_and_latency() -> None:
    rows = [
        {
            "reward": {
                "total": 1.0,
                "format": 1.0,
                "status": 1.0,
                "citation": 1.0,
                "grounding": 1.0,
                "abstention": 1.0,
                "parse_error": None,
            },
            "latency_seconds": 1.0,
        },
        {
            "reward": {
                "total": 0.0,
                "format": 0.0,
                "status": 0.0,
                "citation": 0.0,
                "grounding": 0.0,
                "abstention": 0.0,
                "parse_error": "invalid",
            },
            "latency_seconds": 3.0,
        },
    ]
    report = summarize_prediction_rows(rows)
    assert report["samples"] == 2
    assert report["total"] == pytest.approx(0.5)
    assert report["valid_json_rate"] == pytest.approx(0.5)
    assert report["latency_seconds_mean"] == pytest.approx(2.0)
    assert report["latency_seconds_p95"] == pytest.approx(2.9)


def test_prediction_summary_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize_prediction_rows([])


def test_adapter_evaluation_cli_requires_artifact_and_output() -> None:
    args = build_parser().parse_args(["--adapter-dir", "/tmp/adapter", "--output-dir", "/tmp/eval"])
    assert args.split == "test"
    assert args.max_new_tokens == 512
