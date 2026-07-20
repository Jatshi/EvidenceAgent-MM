# EvidenceAgent-MM v0.1.0

EvidenceAgent-MM v0.1.0 is a reproducible vertical slice for evidence-grounded meeting and classroom QA. It returns claim-level citations, timestamps, anonymous speaker/page provenance, confidence decomposition, and tool traces, and it can ask a targeted clarification or abstain when required evidence is missing.

## Included

- typed multimodal evidence and graph contracts;
- SQLite FTS5 plus deterministic hybrid retrieval;
- bounded Answer / Clarify / Abstain agent;
- FastAPI service and evidence-console web demo;
- CC0 EAMM-Bench Bronze with 12 sessions and 120 questions;
- ablation, calibration, concurrency, and RTX 4090 integration reports;
- optional faster-whisper, BGE-M3, Qwen3-8B, PaddleOCR, and diarization adapters;
- wheel, source distribution, CC0 demo media, benchmark bundle, and SHA-256 checksums.

## Verified results

- 29 tests pass with 88.06% branch-aware coverage.
- Bronze three-state status accuracy and Evidence Recall@5 are 1.000; ECE-10 is 0.413 and intentionally reported as uncalibrated.
- On one AutoDL RTX 4090 smoke input, faster-whisper small yields 2 timestamped segments at WER 0.125, BGE-M3 ranks the target first, Qwen3-8B preserves both required evidence IDs and facts, and PaddleOCR emits 6 uniquely identified atoms from 2 slides.
- The deterministic AutoDL API load smoke completes 200 requests at concurrency 16 with zero failures, 234.5 req/s, and 137.0 ms P95. It excludes GPU model inference.

## Important limits

This release validates contracts and integrations, not real-meeting accuracy. The synthetic ASR output contains a lexical error; mobile OCR omits one numeric phrase; the ungated diarization fallback detects turns rather than persistent speakers; and the confidence score is not calibrated. The demo has no authentication and binds to localhost by default.
