from __future__ import annotations

import json

from evidenceagent_mm.benchmark import generate_bronze, percentile, run_benchmark
from evidenceagent_mm.cli import main


def test_bronze_generator_and_runner(tmp_path) -> None:
    dataset = tmp_path / "bronze"
    summary = generate_bronze(dataset, sessions=2, seed=7)
    assert summary == {"sessions": 2, "questions": 20}
    assert len(list((dataset / "fixtures").glob("*.json"))) == 2
    result = run_benchmark(dataset, tmp_path / "bench.db")
    assert result["metrics"]["questions"] == 20
    assert result["metrics"]["evidence_recall_at_5"] == 1.0
    assert 0 <= result["metrics"]["status_accuracy"] <= 1
    assert len(result["predictions"]) == 20


def test_cli_generates_and_runs_benchmark(tmp_path, capsys) -> None:
    dataset = tmp_path / "cli-bronze"
    assert main(["make-benchmark", str(dataset), "--sessions", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["questions"] == 10
    output = tmp_path / "metrics.json"
    assert (
        main(
            [
                "--db",
                str(tmp_path / "cli.db"),
                "benchmark",
                str(dataset),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["metrics"]["questions"] == 10


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0
