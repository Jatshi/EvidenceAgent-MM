# AutoDL local archive

## Verified scope

The AutoDL export was downloaded to a local archive outside the Git repository and verified against a SHA-256 manifest generated on the remote machine.

```text
VERIFIED files=174 bytes=19201588460
remote manifest SHA-256:
44b1ea1d98bddb135d6e8d65ba589894af13b07b38bc5af8e1db396d68eed5db
```

The public Git repository never stores the private local archive path, server address, password, or authentication tokens.

## Included

- AutoDL source snapshot and source tarball;
- all GPU and system integration reports;
- complete synthetic meeting media, individual speaker WAV files, slides, and manifest;
- `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`;
- `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181`;
- `Systran/faster-whisper-small` revision `536b0662742c02347bc0e980a01041f333bce120`;
- PaddleOCR `PP-OCRv5_mobile_det` and `PP-OCRv5_mobile_rec` inference files;
- core/GPU/OCR dependency freezes, kernel information, and full `nvidia-smi -q` output;
- the final 174-file remote SHA-256 manifest.

## Intentionally excluded

The `.venv`, `gpu-venv`, and `ocr-venv` directories are not copied. They contain Linux-specific interpreter paths, binaries, symlinks, and compiled extensions and cannot be reused as Windows environments. Keeping them would consume about 11.6 GB without improving reproducibility. Exact package freezes and the repository installation scripts are included instead.

An unused older BGE cache revision is also excluded. The archive contains only the exact revision used by the published experiment.

## Layout

```text
autodl-2026-07-20/
├── artifacts/
│   └── export-2026-07-20/
│       ├── core-requirements-freeze.txt
│       ├── gpu-requirements-freeze.txt
│       ├── ocr-requirements-freeze.txt
│       ├── nvidia-smi-q.txt
│       └── sha256-archive-final.txt
├── data/demo_meeting_v2/
├── models/
│   ├── Qwen--Qwen3-8B/<revision>/
│   ├── BAAI--bge-m3/<revision>/
│   ├── Systran--faster-whisper-small/<revision>/
│   └── PaddleOCR/
├── repo-6ea117d/
├── results/
└── source.tgz
```

## Verify a copy

```bash
python scripts/verify_autodl_archive.py /path/to/autodl-2026-07-20
```

The verifier maps audited remote paths to the portable archive layout, hashes every file in 16 MiB chunks, rejects paths outside the known export roots, and exits non-zero on a missing file or mismatch.

## Third-party model handling

The local archive preserves the exact upstream snapshots used for offline reproducibility. They are not committed to GitHub and are not relabeled as EvidenceAgent-MM weights. The [Hugging Face system repository](https://huggingface.co/jatshi/EvidenceAgent-MM) publishes attribution, revisions, configuration, result reports, environment freezes, and the archive hash manifest while leaving checkpoints under their authoritative upstream model repositories.
