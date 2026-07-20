"""Synthetic, redistributable EAMM-Bench Bronze generator and runner."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from evidenceagent_mm.agent import EvidenceAgent
from evidenceagent_mm.evaluation import brier_score, expected_calibration_error, retrieval_metrics
from evidenceagent_mm.pipeline import ingest_fixture
from evidenceagent_mm.retrieval import HashingEncoder, HybridRetriever
from evidenceagent_mm.schema import EvidenceAtom, EvidenceEdge, FixtureDocument, Modality
from evidenceagent_mm.store import EvidenceStore


def generate_bronze(output_dir: str | Path, sessions: int = 12, seed: int = 7) -> dict[str, int]:
    if sessions < 1:
        raise ValueError("sessions must be positive")
    root = Path(output_dir)
    fixture_dir = root / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    question_rows: list[dict[str, Any]] = []
    for session_index in range(sessions):
        session_id = f"eamm-{session_index:03d}"
        speaker_a = f"SPEAKER_{session_index % 3:02d}"
        speaker_b = f"SPEAKER_{(session_index + 1) % 3:02d}"
        page = 5 + session_index
        topic = f"方案-{session_index:02d}"
        atoms = [
            EvidenceAtom(
                evidence_id=f"{session_id}:utt:01",
                session_id=session_id,
                modality=Modality.TRANSCRIPT,
                start_ms=10_000,
                end_ms=14_000,
                speaker_id=speaker_a,
                text=f"我提议采用{topic}，因为它把检索延迟降低到 {40 + session_index} 毫秒。",
                source_uri=f"media://{session_id}.mp4#t=10,14",
                confidence=0.96,
            ),
            EvidenceAtom(
                evidence_id=f"{session_id}:ocr:01",
                session_id=session_id,
                modality=Modality.OCR,
                start_ms=9_000,
                end_ms=18_000,
                page_no=page,
                text=f"{topic}：P95 延迟 {40 + session_index} ms",
                source_uri=f"image://{session_id}-slide-{page:02d}.png",
                confidence=0.94,
            ),
            EvidenceAtom(
                evidence_id=f"{session_id}:utt:02",
                session_id=session_id,
                modality=Modality.TRANSCRIPT,
                start_ms=21_000,
                end_ms=25_000,
                speaker_id=speaker_b,
                text="我建议下周再讨论预算和交付风险。",
                source_uri=f"media://{session_id}.mp4#t=21,25",
                confidence=0.95,
            ),
        ]
        fixture = FixtureDocument(
            session_id=session_id,
            title=f"Synthetic meeting {session_index:03d}",
            duration_ms=30_000,
            source_license="CC0-1.0",
            atoms=atoms,
            edges=[
                EvidenceEdge(
                    source_id=atoms[0].evidence_id,
                    target_id=atoms[1].evidence_id,
                    relation="shown_during",
                    confidence=0.94,
                )
            ],
        )
        (fixture_dir / f"{session_id}.json").write_text(
            json.dumps(fixture.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        answer_question = f"谁提出了{topic}，当时屏幕上是第几页？"
        for repeat in range(6):
            question_rows.append(
                {
                    "question_id": f"{session_id}:answer:{repeat}",
                    "session_id": session_id,
                    "question": answer_question,
                    "expected_status": "answered",
                    "gold_evidence_ids": [atoms[0].evidence_id, atoms[1].evidence_id],
                }
            )
        for repeat in range(2):
            question_rows.append(
                {
                    "question_id": f"{session_id}:clarify:{repeat}",
                    "session_id": session_id,
                    "question": "他提出的方案怎么样？",
                    "expected_status": "needs_clarification",
                    "gold_evidence_ids": [],
                }
            )
        for repeat in range(2):
            question_rows.append(
                {
                    "question_id": f"{session_id}:abstain:{repeat}",
                    "session_id": session_id,
                    "question": "法务最终批准了哪一份合同？",
                    "expected_status": "abstained",
                    "gold_evidence_ids": [],
                }
            )
    (root / "questions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in question_rows),
        encoding="utf-8",
    )
    manifest = {
        "name": "EAMM-Bench Bronze",
        "version": "0.1.0",
        "license": "CC0-1.0",
        "seed": seed,
        "sessions": sessions,
        "questions": len(question_rows),
        "raw_media_redistributable": True,
        "synthetic": True,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"sessions": sessions, "questions": len(question_rows)}


def run_benchmark(dataset_dir: str | Path, db_path: str | Path) -> dict[str, Any]:
    root = Path(dataset_dir)
    rows = [
        json.loads(line)
        for line in (root / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    store = EvidenceStore(db_path)
    try:
        for fixture_path in sorted((root / "fixtures").glob("*.json")):
            fixture = FixtureDocument.model_validate_json(fixture_path.read_text(encoding="utf-8"))
            ingest_fixture(store, fixture)
        retriever = HybridRetriever(store, HashingEncoder(384))
        agent = EvidenceAgent(retriever)
        predictions: list[dict[str, Any]] = []
        latencies: list[float] = []
        retrieval_recalls: list[float] = []
        status_correct: list[int] = []
        probabilities: list[float] = []
        for row in rows:
            started = time.perf_counter()
            hits = retriever.search(row["session_id"], row["question"], top_k=5)
            response = agent.answer(row["session_id"], row["question"], top_k=5)
            latency = (time.perf_counter() - started) * 1_000
            latencies.append(latency)
            gold = set(row["gold_evidence_ids"])
            if gold:
                retrieval_recalls.append(
                    retrieval_metrics([hit.atom.evidence_id for hit in hits], gold, k=5)[
                        "recall_at_k"
                    ]
                )
            correct = int(response.status.value == row["expected_status"])
            status_correct.append(correct)
            probabilities.append(response.confidence.calibrated_overall)
            predictions.append(
                {
                    "question_id": row["question_id"],
                    "expected_status": row["expected_status"],
                    "predicted_status": response.status.value,
                    "correct": bool(correct),
                    "confidence": response.confidence.calibrated_overall,
                    "evidence_ids": [citation.evidence_id for citation in response.citations],
                    "latency_ms": latency,
                }
            )
        metrics = {
            "questions": len(rows),
            "status_accuracy": statistics.mean(status_correct),
            "evidence_recall_at_5": statistics.mean(retrieval_recalls),
            "latency_ms_mean": statistics.mean(latencies),
            "latency_ms_p95": percentile(latencies, 95),
            "brier": brier_score(probabilities, status_correct),
            "ece_10": expected_calibration_error(probabilities, status_correct, bins=10),
        }
        return {"metrics": metrics, "predictions": predictions}
    finally:
        store.close()


def percentile(values: list[float], value: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * value / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
