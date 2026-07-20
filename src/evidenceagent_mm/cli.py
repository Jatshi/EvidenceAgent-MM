"""Command-line entry point for fixtures, queries, benchmark, and serving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="eamm")
    root.add_argument("--db", default="data/processed/evidence.db")
    commands = root.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest-fixture")
    ingest.add_argument("path")
    query = commands.add_parser("query")
    query.add_argument("session_id")
    query.add_argument("question")
    generate = commands.add_parser("make-benchmark")
    generate.add_argument("output_dir")
    generate.add_argument("--sessions", type=int, default=12)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("dataset_dir")
    benchmark.add_argument("--output", default="results/exp_001/metrics.json")
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "make-benchmark":
        from evidenceagent_mm.benchmark import generate_bronze

        print(json.dumps(generate_bronze(args.output_dir, args.sessions), ensure_ascii=False))
        return 0
    if args.command == "benchmark":
        from evidenceagent_mm.benchmark import run_benchmark

        result = run_benchmark(args.dataset_dir, args.db)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result["metrics"], ensure_ascii=False))
        return 0
    if args.command == "serve":
        import uvicorn

        from evidenceagent_mm.api import create_app

        uvicorn.run(create_app(args.db), host=args.host, port=args.port)
        return 0

    from evidenceagent_mm.pipeline import ingest_fixture_file
    from evidenceagent_mm.store import EvidenceStore

    with EvidenceStore(args.db) as store:
        if args.command == "ingest-fixture":
            print(json.dumps(ingest_fixture_file(store, args.path), ensure_ascii=False))
            return 0
        if args.command == "query":
            from evidenceagent_mm.agent import EvidenceAgent
            from evidenceagent_mm.retrieval import HybridRetriever

            response = EvidenceAgent(HybridRetriever(store)).answer(args.session_id, args.question)
            print(response.model_dump_json(indent=2))
            return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
