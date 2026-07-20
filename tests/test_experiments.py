from __future__ import annotations

from evidenceagent_mm.benchmark import generate_bronze
from evidenceagent_mm.experiments import run_ablation_suite


def test_ablation_suite_reports_all_configs(tmp_path) -> None:
    dataset = tmp_path / "bronze"
    generate_bronze(dataset, sessions=1)
    report = run_ablation_suite(dataset, tmp_path / "work")
    assert set(report["configs"]) == {"full", "no_graph", "top1", "no_visual_gate"}
    assert report["configs"]["full"]["status_accuracy"] == 1.0
    assert report["configs"]["full"]["negative_false_answer_rate"] == 0.0
    assert report["configs"]["top1"]["status_accuracy"] < 1.0
