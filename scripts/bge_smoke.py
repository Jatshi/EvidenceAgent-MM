"""Measure BGE-M3 retrieval on a tiny cross-lingual fixture."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

from evidenceagent_mm.provenance import huggingface_cache_revision
from evidenceagent_mm.retrieval import SentenceTransformerEncoder


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--output", default="results/gpu/bge_smoke.json")
    args = parser.parse_args()
    corpus = [
        "SPEAKER_00 proposed design B to reduce retrieval latency.",
        "SPEAKER_01 requested a budget review next week.",
        "The classroom projector was turned off.",
    ]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    encoder = SentenceTransformerEncoder(args.model, device="cuda")
    loaded = time.perf_counter()
    query = encoder.encode(["谁提出了降低检索延迟的方案？"])[0]
    vectors = encoder.encode(corpus)
    scores = (vectors @ query).tolist()
    finished = time.perf_counter()
    result = {
        "model": args.model,
        "model_revision": huggingface_cache_revision(args.model),
        "scores": scores,
        "top_index": max(range(len(scores)), key=scores.__getitem__),
        "model_load_seconds": loaded - started,
        "encode_seconds": finished - loaded,
        "elapsed_seconds": finished - started,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return int(result["top_index"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
