# EvidenceAgent-MM 2.0 implementation and execution plan

## 2.0 objective

Turn the v0.1 evidence-grounded assistant into a reproducible post-trained
multimodal Agent while retaining the deterministic state machine as the safety
baseline and preserving every existing `/v1` API route.

The implementation is complete locally when data contracts, deterministic
builders, rewards, feedback export, training entry points, AutoDL scripts, tests,
lint, and dry-runs pass. The release is empirically complete only after the AutoDL
smoke/full runs create model adapters, logs, metrics, and manifests.

## 2.0 acceptance ledger

- [x] Pydantic `2.0` contracts for SFT, DPO, GRPO, target JSON, metadata, and weights.
- [x] Deterministic Bronze-to-training builder with session-disjoint splits.
- [x] ASR/acoustic adapter carrying backend, model, diagnostics, SNR, overlap, speech
  probability, noise type, and language into `EvidenceAtom.attributes`.
- [x] Evidence prompt serialization preserving timestamp, modality, speaker, page,
  source URI, confidence, ASR, and acoustic provenance.
- [x] Offline-verifiable format/status/citation/grounding/abstention reward.
- [x] TRL-compatible batched reward wrapper and unit tests including hallucinated
  citations and safe abstention.
- [x] Append-only hard-case store with stable IDs, deduplication, explicit consent,
  correction pairs, and SFT/DPO/GRPO export.
- [x] Optional SFT, DPO, and GRPO LoRA runtime with dependency-light dry-run.
- [x] Conservative Qwen3-1.7B single-4090 configs and one-step smoke mode.
- [x] AutoDL bootstrap, GPU/disk/dependency/config preflight, staged run, logs, and
  JSON run manifests.
- [x] Optional DeepSpeed dependency, checked ZeRO-2/ZeRO-3 CPU-offload configs,
  Trainer passthrough, launcher selection, and dependency-light dry-run.
- [x] v0.1 deterministic Agent and `/v1` API compatibility retained.
- [ ] AutoDL 4090 smoke: one optimizer step for SFT, DPO, and GRPO.
- [ ] AutoDL full training and held-out evaluation.
- [ ] Real multi-GPU DeepSpeed smoke with `world_size>=2`, per-rank launch evidence,
  completed manifest, and reloadable adapter.
- [ ] Publish only the project's trained adapters and their model cards to Hugging Face.

## Execution sequence

1. Copy the repository to AutoDL without virtual environments or model caches.
2. Run `bash scripts/autodl_v2_bootstrap.sh`.
3. Run `bash scripts/autodl_v2_preflight.sh`; stop if GPU, disk, dependencies, data,
   or configs fail.
4. Run `PORTFOLIO_V2_MODE=smoke bash scripts/autodl_v2_run.sh`.
5. Inspect each stage's `run_manifest.json`, loss, CUDA errors, and peak VRAM.
6. Run `PORTFOLIO_V2_MODE=full bash scripts/autodl_v2_run.sh`.
7. Evaluate held-out sessions and compare deterministic, SFT, DPO, and GRPO systems.
8. Export hard cases only when license and training consent are explicit.
9. Publish adapters only after checksums, model cards, limitations, and metrics exist.

## Required experiment matrix

| Run | Model | Training | Primary evidence |
|---|---|---|---|
| baseline | deterministic v0.1 | none | status/citation contract |
| SFT | Qwen3-1.7B + LoRA | target JSON | schema validity and task accuracy |
| DPO | Qwen3-1.7B + LoRA | chosen/rejected | citation and refusal preference |
| GRPO | Qwen3-1.7B + LoRA | five deterministic rewards | grounding and selective risk |

Report status accuracy, citation precision/recall/F1, unsupported-claim rate,
abstention precision/recall, answer coverage, selective risk, ECE, latency, and
peak VRAM. Do not infer real-world quality from Bronze alone.

## AutoDL cost controls

- Default `PORTFOLIO_V2_MODE=smoke`; full training requires an explicit environment value.
- Qwen3-1.7B, LoRA rank 16, batch size 1, gradient accumulation, and 2,048-token cap.
- SFT/DPO use NF4; GRPO uses BF16 because quantized GRPO compatibility varies across
  TRL/bitsandbytes versions, but the 1.7B model remains conservative for 24 GiB.
- Checkpoints are capped at two per stage and saved under `/root/autodl-tmp/eamm-v2`
  by default.
- Preflight requires 20 GiB VRAM and 35 GiB free disk.
- DeepSpeed `world_size=1` is a compatibility check only; scaling and distributed
  correctness require a future machine with at least two GPUs.

## Fact boundary

As of 2026-07-28, all local contracts, code paths, and dry-runs can be verified
without a remote GPU. No 2.0 loss curve, learned improvement, final adapter,
throughput, or peak-VRAM number may be placed on a resume until the corresponding
AutoDL artifact exists.

---

# v0.1 implementation ledger

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
- [x] Full/no-graph/top-1/no-visual-gate ablation suite with explicit negative findings.
- [x] Base unit/API tests, Ruff, coverage, CI, packaging, and security documentation.
- [x] Local API concurrency smoke with failure rate, throughput, mean, P50, P95, and max.
- [x] AutoDL synthetic media generated and archived with SHA-256.
- [x] faster-whisper ASR measured on RTX 4090.
- [x] BGE-M3 retrieval measured on RTX 4090.
- [x] Qwen3-8B constrained generation measured on RTX 4090.
- [x] OCR adapter measured on the generated slides.
- [x] Diarization path measured with Community-1 or documented Plan B.
- [x] Release artifacts, exact dependency freeze, GitHub repository, and v0.1.0 release published.

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
