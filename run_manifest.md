# Run manifest

## CPU contract run

- Date: 2026-07-20
- Command: `make benchmark`
- Python: 3.11.7
- Dataset: EAMM-Bench Bronze v0.1.0, 12 sessions, 120 questions, CC0
- Output: `benchmarks/results/cpu_bronze.json`
- Status: verified
- Scope: software contract and evidence accounting only

## AutoDL GPU integration

The authoritative GPU reports are stored under `benchmarks/results/gpu/`.

- Environment: RTX 4090, driver 570.124.04, PyTorch 2.10.0+cu128, Python 3.10.8.
- ASR: `scripts/gpu_asr_smoke.py`, faster-whisper small revision `536b066...`, WER 0.125, cached run 4.42 s.
- Retrieval: `scripts/bge_smoke.py`, BGE-M3 revision `5617a9f...`, correct cross-lingual top-1, cached run 25.17 s.
- Qwen3, OCR, and diarization reports are added only after their integration gates complete.

Each report must contain GPU name, driver, model name/revision, runtime, output, and the generating command. Missing reports mean the integration has not been verified.
