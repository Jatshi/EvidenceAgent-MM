# Changelog

## 3.0.0 - 2026-08-13

- Added budgeted, replayable evidence search and claim-verification tool trajectories.
- Added prompt-injection boundaries and deterministic hard cases for answer, clarify, and abstain.
- Added VERL/GRPO trajectory training with dense rewards and non-zero-gradient gates.
- Completed 50 RTX 4090 actor updates and exported a CUDA-validated 34MB PEFT adapter.
- Added separate 3.0 release notes and a deep Chinese learning, failure, and interview guide.

## 2.0.0 - prepared 2026-07-28

- Add versioned SFT, DPO, and GRPO contracts and deterministic builders.
- Preserve ASR and acoustic conditions in evidence-grounded training prompts.
- Add verifiable citation, status, grounding, and abstention rewards.
- Add consent-gated hard-case feedback and data-flywheel export.
- Add optional single-4090 TRL LoRA training, dry-runs, and AutoDL orchestration.
- Add optional DeepSpeed ZeRO-2/ZeRO-3 CPU-offload configurations, launcher
  selection, validation, and explicit single-process versus multi-GPU evidence boundaries.
- Preserve the deterministic Agent and all existing `/v1` routes.

All notable changes follow Keep a Changelog and Semantic Versioning.

## [Unreleased]

- Real-meeting benchmark expansion and validation-set calibration.
- Published the EvidenceAgent-MM system model card, configuration, raw results, and provenance on Hugging Face.
- Added a SHA-256 AutoDL archive verifier and documented the 174-file verified local export.
- Added a comprehensive Chinese from-scratch tutorial with formulas, runnable core code, source walkthrough, rebuild labs, and 25 interview questions.

## [0.1.0] - 2026-07-20

### Added

- Typed multimodal evidence schema and bounded evidence graph.
- SQLite FTS5 plus deterministic CJK-aware dense/lexical hybrid retrieval.
- Answer, clarification, and abstention state contract.
- Claim-level citations, tool traces, confidence decomposition, and missing-evidence taxonomy.
- FastAPI service and responsive evidence-report demo.
- CC0 EAMM-Bench Bronze generator with 12 sessions and 120 questions.
- Calibration, retrieval, citation, and selective answering metrics.
- Optional faster-whisper, BGE-M3, PaddleOCR, pyannote, and Qwen3 adapters.
- Unit/API benchmark suite, coverage gate, Ruff, pre-commit, and GitHub Actions.
- Reproducible RTX 4090 integration reports with exact model revisions and runtime provenance.
- Stable image- and timestamp-derived OCR evidence IDs with cross-slide collision regression coverage.
- Polished evidence-console screenshot and redistributable synthetic demo media release asset.
- NumPy `<2.3` compatibility bound so Mypy can validate the declared Python 3.10 target on newer CI interpreters.
