from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from evidenceagent_mm.perception import EnergyTurnDetector, ocr_evidence_id


def test_energy_turn_detector_separates_long_silence(tmp_path) -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    first = (4_000 * np.sin(2 * np.pi * 180 * time)).astype(np.int16)
    silence = np.zeros(sample_rate, dtype=np.int16)
    second = (4_000 * np.sin(2 * np.pi * 260 * time)).astype(np.int16)
    samples = np.concatenate([first, silence, second])
    path = tmp_path / "turns.wav"
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(samples.tobytes())

    atoms = EnergyTurnDetector().detect(path, "test-session")
    assert len(atoms) == 2
    assert atoms[0].speaker_id == "SPEAKER_00"
    assert atoms[1].start_ms >= 1_900


def test_ocr_evidence_id_is_idempotent_and_image_specific(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first-image")
    second.write_bytes(b"second-image")

    evidence_id = ocr_evidence_id("session", first, 7_000, 0, 1)
    assert evidence_id == ocr_evidence_id("session", first, 7_000, 0, 1)
    assert evidence_id != ocr_evidence_id("session", second, 7_000, 0, 1)
    assert evidence_id != ocr_evidence_id("session", first, 8_000, 0, 1)
