# EvidenceAgent-MM implementation ledger

## Goal and acceptance

Build a reproducible meeting assistant whose claims are traceable to timestamped multimodal evidence and whose unsupported questions trigger clarification or abstention.

Release acceptance requires: core tests and coverage gate pass; Bronze benchmark artifacts are reproducible; each answered claim cites evidence; the three states are demonstrated; GPU scripts record model/device revisions; private media and credentials are absent from Git; the GitHub release contains source, wheel, dataset card, model card, and checksums.

## Vertical slices

- [x] Typed `EvidenceAtom`, edge, claim, citation, response, and fixture contracts.
- [x] Evidence graph construction and bounded expansion.
- [x] Parameterized SQLite store, FTS5, CJK-aware token overlap, dense baseline, and RRF.
- [x] Three-state deterministic tool agent and evidence sufficiency gate.
- [x] FastAPI routes and evidence-report web demo.
- [x] EAMM-Bench Bronze generator, 120-question run, calibration metrics, and predictions.
- [x] Base unit/API tests, Ruff, coverage, CI, packaging, and security documentation.
- [ ] AutoDL synthetic media generated and archived with SHA-256.
- [ ] faster-whisper ASR measured on RTX 4090.
- [ ] BGE-M3 retrieval measured on RTX 4090.
- [ ] Qwen3-8B constrained generation measured on RTX 4090.
- [ ] OCR adapter measured on the generated slides.
- [ ] Diarization path measured with Community-1 or documented Plan B.
- [ ] Release artifacts, exact dependency freeze, GitHub repository, and v0.1.0 release published.

## Decision log

| Decision | Choice | Reason |
|---|---|---|
| Core database | SQLite/FTS5 baseline | offline tests and zero service dependency; pgvector remains production scale path |
| Core embedding | deterministic hashing | contract tests cannot silently download models |
| Production embedding | BGE-M3 | multilingual and long-context model with an explicit upstream card |
| Agent default | fixed bounded state machine | traceable baseline before generative planning |
| Generation | injectable Qwen3 adapter | generation cannot bypass evidence selection/gating |
| Benchmark media | synthetic CC0 first | fully redistributable, exact gold timing, no privacy risk |

## Known limitations

- Synthetic status accuracy is not evidence of real-world accuracy.
- Confidence output is a baseline score, not yet a calibrated probability.
- Pyannote Community-1 access depends on upstream gated-model acceptance.
- v0.1 Demo has no authentication and is restricted to localhost.
