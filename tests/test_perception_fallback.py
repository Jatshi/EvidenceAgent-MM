from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from evidenceagent_mm.perception import (
    AcousticFeatures,
    EnergyTurnDetector,
    asr_segment_to_evidence,
    ocr_evidence_id,
)


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


def test_asr_segment_adapter_preserves_acoustic_metadata() -> None:
    atom = asr_segment_to_evidence(
        session_id="session",
        media_path="meeting.wav",
        index=2,
        start_seconds=1.0,
        end_seconds=1.0,
        text="robust speech evidence",
        confidence=1.2,
        backend="test-asr",
        model_name="tiny",
        speaker_id="SPEAKER_00",
        acoustic=AcousticFeatures(
            snr_db=-2.0,
            overlap_probability=0.7,
            speech_probability=0.9,
            noise_type="babble",
            language="en",
        ),
        diagnostics={"avg_logprob": -0.2},
    )

    assert atom.end_ms == atom.start_ms + 1
    assert atom.confidence == 1.0
    assert atom.attributes["asr"]["backend"] == "test-asr"
    assert atom.attributes["acoustic"]["snr_db"] == -2.0
