# EAMM-Bench Bronze dataset card

## Summary

EAMM-Bench Bronze v0.1.0 is a synthetic CC0 contract benchmark for evidence-grounded meeting QA. It contains 12 sessions and 120 questions: 72 answerable, 24 clarifiable, and 24 unanswerable.

Each session contains two anonymous speakers, one proposal utterance, one overlapping OCR/slide atom, and one unrelated follow-up utterance. Gold evidence IDs make retrieval and claim citation measurable.

## Intended use

- regression tests for ingestion, graph construction, retrieval, and three-state behavior;
- tutorial data that can be redistributed without privacy review;
- deterministic CI and API demonstrations.

## Out-of-scope use

The benchmark must not be used to claim real meeting accuracy, speaker robustness, OCR quality, ASR quality, demographic fairness, or production calibration. Its repeated templates deliberately make it easy enough to expose software regressions.

## Generation and split policy

`eamm make-benchmark` deterministically creates the metadata. A later Silver release will use session-level train/validation/test splits, annotation double-coding, acoustic perturbations, and real corpora whose licenses permit the chosen distribution method.

## License and privacy

The generated benchmark metadata is CC0-1.0. It contains no real voices, faces, names, or private meetings. Upstream corpora such as AMI, QMSum, or AISHELL-4 are never bundled here and retain their own terms.
