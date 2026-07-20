"""Reproducible ablation runner over an EAMM benchmark directory."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from evidenceagent_mm.agent import EvidenceAgent, GateConfig
from evidenceagent_mm.evaluation import retrieval_metrics
from evidenceagent_mm.pipeline import ingest_fixture
from evidenceagent_mm.retrieval import HashingEncoder, HybridRetriever
from evidenceagent_mm.schema import FixtureDocument, ResponseStatus
from evidenceagent_mm.store import EvidenceStore


def run_ablation_suite(dataset_dir: str | Path, work_dir: str | Path) -> dict[str, Any]:
    dataset = Path(dataset_dir)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in (dataset / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    configs = {
        "full": {"top_k": 5, "gate": GateConfig(graph_hops=1)},
        "no_graph": {"top_k": 5, "gate": GateConfig(graph_hops=0)},
        "top1": {"top_k": 1, "gate": GateConfig(graph_hops=0)},
        "no_visual_gate": {
            "top_k": 5,
            "gate": GateConfig(graph_hops=1, require_cross_modal_for_visual_questions=False),
        },
    }
    reports: dict[str, Any] = {}
    for name, values in configs.items():
        db_path = work / f"{name}.db"
        if db_path.exists():
            db_path.unlink()
        with EvidenceStore(db_path) as store:
            for fixture_path in sorted((dataset / "fixtures").glob("*.json")):
                ingest_fixture(
                    store,
                    FixtureDocument.model_validate_json(fixture_path.read_text(encoding="utf-8")),
                )
            retriever = HybridRetriever(store, HashingEncoder(384))
            agent = EvidenceAgent(retriever, config=values["gate"])
            status_correct: list[int] = []
            recalls: list[float] = []
            negative_answers = 0
            negative_total = 0
            started = time.perf_counter()
            for row in rows:
                top_k = int(values["top_k"])
                response = agent.answer(row["session_id"], row["question"], top_k=top_k)
                status_correct.append(int(response.status.value == row["expected_status"]))
                if row["gold_evidence_ids"]:
                    hits = retriever.search(
                        row["session_id"],
                        row["question"],
                        top_k=top_k,
                        graph_hops=values["gate"].graph_hops,
                    )
                    recalls.append(
                        retrieval_metrics(
                            [hit.atom.evidence_id for hit in hits],
                            set(row["gold_evidence_ids"]),
                            k=5,
                        )["recall_at_k"]
                    )
                else:
                    negative_total += 1
                    negative_answers += int(response.status is ResponseStatus.ANSWERED)
            reports[name] = {
                "top_k": values["top_k"],
                "graph_hops": values["gate"].graph_hops,
                "require_cross_modal_for_visual_questions": values[
                    "gate"
                ].require_cross_modal_for_visual_questions,
                "questions": len(rows),
                "status_accuracy": statistics.mean(status_correct),
                "evidence_recall_at_5": statistics.mean(recalls),
                "negative_false_answer_rate": negative_answers / negative_total,
                "elapsed_seconds": time.perf_counter() - started,
            }
    return {"dataset": dataset.as_posix(), "configs": reports}
