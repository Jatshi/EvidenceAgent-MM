# Artifact layout

- `README.md`: Hugging Face system model card.
- `system_config.json`: machine-readable component revisions and gate thresholds.
- `results/`: committed CPU/GPU integration and benchmark reports.
- `provenance/`: dependency freezes, environment inventory, and the SHA-256 manifest used to verify the local AutoDL archive.
- `LICENSE`: Apache-2.0 license for EvidenceAgent-MM code and system configuration.

No third-party checkpoint is redistributed here. The SHA manifest records the exact files used locally, while the model card links the authoritative upstream repositories and revisions.
