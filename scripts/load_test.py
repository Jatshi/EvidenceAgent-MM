"""Small dependency-light concurrent load test for the local query API."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from evidenceagent_mm.evaluation import percentile


def request_once(url: str, payload: dict[str, object]) -> tuple[int, float]:
    started = time.perf_counter()
    response = httpx.post(url, json=payload, timeout=20)
    return response.status_code, (time.perf_counter() - started) * 1_000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/query")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--output", default="results/system/api_load.json")
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        raise SystemExit("requests and concurrency must be positive")
    payload = {
        "session_id": "eamm-000",
        "question": "谁提出了方案-00，当时屏幕上是第几页？",
        "top_k": 5,
    }
    wall_started = time.perf_counter()
    results: list[tuple[int, float]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(request_once, args.url, payload) for _ in range(args.requests)]
        for future in as_completed(futures):
            results.append(future.result())
    wall_seconds = time.perf_counter() - wall_started
    latencies = [latency for _, latency in results]
    failures = sum(status != 200 for status, _ in results)
    report = {
        "url": args.url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "failures": failures,
        "failure_rate": failures / args.requests,
        "throughput_rps": args.requests / wall_seconds,
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "latency_ms_max": max(latencies),
        "wall_seconds": wall_seconds,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
