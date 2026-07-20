---
language:
  - en
  - zh
license: apache-2.0
library_name: custom
pipeline_tag: question-answering
tags:
  - multimodal
  - rag
  - agent
  - meeting-assistant
  - evidence-grounding
  - citation
  - abstention
  - reproducibility
---

# EvidenceAgent-MM v0.1.0

EvidenceAgent-MM is an evidence-grounded multimodal system for noisy meetings and classrooms. It answers questions such as *who proposed what, when, why, and which slide was visible* with claim-level citations, timestamps, anonymous speaker/page provenance, confidence decomposition, and a tool trace. When evidence is ambiguous or insufficient, the system asks a targeted clarification or abstains.

> This repository is a **system model card and reproducibility artifact**, not a newly trained neural checkpoint. EvidenceAgent-MM v0.1.0 composes pinned upstream models behind a deterministic evidence gate. No upstream weights are copied into this repository and no fine-tuning is claimed.

- Source: <https://github.com/Jatshi/EvidenceAgent-MM>
- Release: <https://github.com/Jatshi/EvidenceAgent-MM/releases/tag/v0.1.0>
- Chinese guide: <https://github.com/Jatshi/EvidenceAgent-MM/blob/main/README.zh-CN.md>
- From-scratch Chinese tutorial: <https://github.com/Jatshi/EvidenceAgent-MM/blob/main/docs/tutorials/evidenceagent_mm_from_scratch_tutorial.md>

## System composition

| Role | Upstream component | Exact revision / model | Verified use |
|---|---|---|---|
| Generation | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | evidence-constrained wording after gating |
| Dense retrieval | `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` | multilingual evidence ranking |
| ASR | `Systran/faster-whisper-small` | `536b0662742c02347bc0e980a01041f333bce120` | timestamped transcript atoms |
| OCR | PaddleOCR 3.7 | `PP-OCRv5_mobile_det` + `PP-OCRv5_mobile_rec` | screen/slide evidence atoms |
| Core gate | EvidenceAgent-MM | v0.1.0 | Answer / Clarify / Abstain control flow |

Consult each upstream repository for its own model license and usage restrictions. EvidenceAgent-MM code and the system-level configuration in this repository are Apache-2.0.

## What the “model” is

The core contract is not “send a video to an LLM.” It is:

1. ASR, OCR, and turn detection create typed `EvidenceAtom` objects.
2. SQLite FTS5, deterministic/dense embeddings, reciprocal-rank fusion, and a bounded graph retrieve evidence.
3. A deterministic gate checks ambiguity, required modalities, retrieval quality, and claim support.
4. Only evidence that passes the gate reaches the optional Qwen renderer.
5. The response is one of `answered`, `needs_clarification`, or `abstained`.

The exact thresholds and pinned components are machine-readable in [`system_config.json`](system_config.json).

## Verified results

All measurements below use one CC0 synthetic 12.4-second meeting unless explicitly marked otherwise. They validate integration and contracts, not real-meeting accuracy.

| Check | Result |
|---|---:|
| EAMM-Bench Bronze | 12 sessions / 120 questions |
| Three-state status accuracy | 1.000 |
| Evidence Recall@5 | 1.000 |
| ECE-10 | 0.413, uncalibrated |
| Core tests | 29 passed, 88.08% branch-aware coverage |
| faster-whisper small | 2 segments, WER 0.125, 1.587 s warm-cache |
| BGE-M3 | correct target at rank 1, score 0.625, 7.710 s |
| Qwen3-8B | both citations and required facts preserved, 12.201 s total |
| Qwen3-8B peak VRAM | 15,665 MiB |
| PaddleOCR | 6 atoms from 2 slides, 6 unique IDs, 2.461 s |
| Turn-detection fallback | 2/2 turns, temporal IoU 0.914; not speaker identity |
| AutoDL deterministic API | 200 requests, 0 failures, 234.5 req/s, P95 137.0 ms |

Raw JSON reports are stored in [`results/`](results/). Exact environment freezes and SHA-256 provenance are stored in [`provenance/`](provenance/).

## Quick start

```bash
git clone https://github.com/Jatshi/EvidenceAgent-MM.git
cd EvidenceAgent-MM
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
make benchmark
eamm --db /tmp/eamm.db serve --host 127.0.0.1 --port 8000
```

The web demo has no authentication. Keep it on localhost unless you add an authenticated reverse proxy and a retention policy for meeting media.

To download this Hub artifact itself:

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="jatshi/EvidenceAgent-MM", local_dir="EvidenceAgent-MM-hub")
```

The pinned upstream models can be downloaded independently with their exact revisions. This preserves attribution, lets the Hub deduplicate official files, and avoids presenting third-party checkpoints as EvidenceAgent-MM training output.

## Known limitations

- Bronze is a deliberately easy synthetic contract benchmark and is not evidence of production accuracy.
- Confidence is a hand-built baseline score; ECE-10 shows that it is not calibrated.
- The synthetic ASR output confuses `review` with `renew` and includes a number-format difference.
- Mobile OCR misses `42 ms` and reads `latency` as `Iatency` on the first slide.
- The license-free fallback detects speech turns only. Its sequential labels are not persistent speaker identities.
- Citation presence is not the same as semantic entailment; production use needs claim-level support evaluation on real data.
- Private meeting media requires access control, consent, deletion policies, and an audit log.

## Citation

```bibtex
@software{evidenceagent_mm_2026,
  author = {Shi, Jianting},
  title = {EvidenceAgent-MM: Evidence-Grounded Multimodal Assistance for Noisy Meetings},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/Jatshi/EvidenceAgent-MM}
}
```
