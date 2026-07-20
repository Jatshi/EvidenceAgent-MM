# Run manifest

## CPU contract run

- Date: 2026-07-20
- Command: `make benchmark`
- Python: 3.11.7
- Dataset: EAMM-Bench Bronze v0.1.0, 12 sessions, 120 questions, CC0
- Output: `benchmarks/results/cpu_bronze.json`
- Status: verified
- Scope: software contract and evidence accounting only
- Ablations: top-1 drops recall to 0.5 and status accuracy to 0.2; graph/visual-gate deltas are zero on this small clean fixture and are not claimed as improvements.
- Local API load: 200 requests at concurrency 16, 0 failures, 144.7 req/s, P95 235.8 ms; deterministic CPU path only.

## AutoDL GPU integration

The authoritative GPU reports are stored under `benchmarks/results/gpu/`.

- Environment: RTX 4090, driver 570.124.04, PyTorch 2.10.0+cu128, Python 3.10.8.
- ASR: `scripts/gpu_asr_smoke.py`, faster-whisper small revision `536b0662742c02347bc0e980a01041f333bce120`, 2 segments, WER 0.125, warm-cache run 1.587 s.
- Retrieval: `scripts/bge_smoke.py`, BGE-M3 revision `5617a9f61b028005a4858fdac845db406aefb181`, correct cross-lingual top-1 at 0.625, 7.279 s load plus 0.431 s encode, peak VRAM 2,178 MiB.
- Generation: `scripts/qwen_smoke.py`, Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, both required evidence IDs and facts preserved, 9.618 s load plus 2.583 s generation, peak VRAM 15,665 MiB.
- OCR: `scripts/ocr_smoke.py`, PaddleOCR 3.7.0 / Paddle 3.3.0 on `gpu:0`, PP-OCRv5 mobile detector and recognizer, 6 atoms from 2 slides with 6 unique IDs, warm-cache run 2.461 s. First-slide OCR errors are preserved in the report.
- Diarization Plan B: license-free energy turn detector found 2/2 speech turns with mean temporal IoU 0.914. This is not full speaker diarization; sequential labels are not reusable speaker identities.
- AutoDL API load: 200 deterministic-path requests at concurrency 16, zero failures, 234.5 req/s, P95 137.0 ms; excludes GPU model calls.
- Synthetic media hashes: `benchmarks/results/gpu/demo_media_sha256.txt`; media is distributed as a release asset rather than tracked source.
- OCR runtime provenance: `benchmarks/results/gpu/ocr_environment.json`; Paddle reused the audited CUDA 12.8/cuDNN 9.10 shared libraries from the isolated GPU environment through `LD_LIBRARY_PATH`.

Each report must contain GPU name, driver, model name/revision, runtime, output, and the generating command. Missing reports mean the integration has not been verified.

## Release verification

- Repository: `https://github.com/Jatshi/EvidenceAgent-MM`
- Release: `https://github.com/Jatshi/EvidenceAgent-MM/releases/tag/v0.1.0`
- CI compatibility implementation commit: `c458f08dd19e05f431f5f0fe700f6e6ca22bd7b2`
- Assets: wheel, source distribution, CC0 demo media, benchmark/results bundle, and `SHA256SUMS`.
- CI: Python 3.10 and 3.12 matrix passed Ruff, formatting, Mypy, 29 tests with coverage gate, and package build.

## Post-release preservation and learning artifacts

- Hugging Face system repository: `https://huggingface.co/jatshi/EvidenceAgent-MM`
- Initial Hub artifact commit: `7dff87ec222d667677afcf9608c3ff4ad95e84c2`
- Current Hub artifact commit after linking the learning manual: `769f0e7bb6b213ca0b34ab4a833c22a4669afabb`
- Formatting-normalized Hub provenance commit: `22555d0661957060dc277631c8757bc1f5528946`
- The Hub repository contains no relabeled third-party checkpoint and explicitly declares `trained_weights: false`.
- AutoDL archive verification: 174 files and 19,201,588,460 bytes matched the remote SHA-256 manifest.
- Remote archive manifest SHA-256: `44b1ea1d98bddb135d6e8d65ba589894af13b07b38bc5af8e1db396d68eed5db`.
- Linux virtual environments were excluded as non-portable; exact core/GPU/OCR freezes are retained.
- The complete tutorial source and rendered HTML live under `docs/tutorials/`.
