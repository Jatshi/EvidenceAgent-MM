"""Optional GPU perception adapters with provenance-preserving outputs."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np

from evidenceagent_mm.schema import EvidenceAtom, Modality


class FasterWhisperASR:
    def __init__(
        self, model_size: str = "small", device: str = "cuda", compute_type: str = "float16"
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install evidenceagent-mm[gpu] for ASR") from exc
        self.model: Any = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, media_path: str | Path, session_id: str) -> list[EvidenceAtom]:
        segments, _ = self.model.transcribe(str(media_path), word_timestamps=True, vad_filter=True)
        atoms: list[EvidenceAtom] = []
        for index, segment in enumerate(segments):
            atoms.append(
                EvidenceAtom(
                    evidence_id=f"{session_id}:asr:{index:05d}",
                    session_id=session_id,
                    modality=Modality.TRANSCRIPT,
                    start_ms=max(0, round(segment.start * 1_000)),
                    end_ms=max(1, round(segment.end * 1_000)),
                    text=segment.text.strip(),
                    source_uri=f"media://{Path(media_path).name}#t={segment.start:.3f},{segment.end:.3f}",
                    confidence=max(0.0, min(1.0, float(getattr(segment, "avg_logprob", -1)) + 1)),
                    attributes={"backend": "faster-whisper", "model": "runtime-configured"},
                )
            )
        return atoms


class PaddleOCRAdapter:
    def __init__(self, lang: str = "ch") -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install evidenceagent-mm[ocr] and PaddlePaddle for OCR") from exc
        self.model: Any = PaddleOCR(lang=lang, use_doc_orientation_classify=False)

    def extract(
        self, image_path: str | Path, session_id: str, timestamp_ms: int
    ) -> list[EvidenceAtom]:
        result = self.model.predict(str(image_path))
        atoms: list[EvidenceAtom] = []
        for index, page in enumerate(result):
            data = page.json.get("res", page.json)
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [1.0] * len(texts))
            for line_index, (text, score) in enumerate(zip(texts, scores, strict=False)):
                atoms.append(
                    EvidenceAtom(
                        evidence_id=f"{session_id}:ocr:{index:03d}:{line_index:03d}",
                        session_id=session_id,
                        modality=Modality.OCR,
                        start_ms=timestamp_ms,
                        end_ms=timestamp_ms + 1,
                        text=str(text),
                        source_uri=f"image://{Path(image_path).name}",
                        confidence=float(score),
                        attributes={"backend": "paddleocr"},
                    )
                )
        return atoms


class EnergyTurnDetector:
    """Dependency-light VAD fallback; it detects turns but does not identify speakers."""

    def __init__(self, frame_ms: int = 30, merge_gap_ms: int = 650, min_turn_ms: int = 250) -> None:
        self.frame_ms = frame_ms
        self.merge_gap_ms = merge_gap_ms
        self.min_turn_ms = min_turn_ms

    def detect(self, wav_path: str | Path, session_id: str) -> list[EvidenceAtom]:
        with wave.open(str(wav_path), "rb") as source:
            if source.getnchannels() != 1 or source.getsampwidth() != 2:
                raise ValueError("EnergyTurnDetector expects mono 16-bit PCM WAV")
            sample_rate = source.getframerate()
            samples = np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16).astype(
                np.float32
            )
        frame_size = max(1, round(sample_rate * self.frame_ms / 1_000))
        rms = np.asarray(
            [
                float(np.sqrt(np.mean(samples[start : start + frame_size] ** 2)))
                for start in range(0, len(samples), frame_size)
            ]
        )
        noise_floor = float(np.percentile(rms, 20))
        threshold = max(150.0, noise_floor * 3.0)
        active = np.flatnonzero(rms >= threshold)
        if not len(active):
            return []
        groups: list[list[int]] = [[int(active[0])]]
        allowed_gap = max(1, self.merge_gap_ms // self.frame_ms)
        for frame in active[1:]:
            if int(frame) - groups[-1][-1] <= allowed_gap:
                groups[-1].append(int(frame))
            else:
                groups.append([int(frame)])
        atoms: list[EvidenceAtom] = []
        for index, group in enumerate(groups):
            start_ms = group[0] * self.frame_ms
            end_ms = min(round(len(samples) / sample_rate * 1_000), (group[-1] + 1) * self.frame_ms)
            if end_ms - start_ms < self.min_turn_ms:
                continue
            atoms.append(
                EvidenceAtom(
                    evidence_id=f"{session_id}:vad:{index:03d}",
                    session_id=session_id,
                    modality=Modality.AUDIO,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    speaker_id=f"SPEAKER_{index:02d}",
                    text="",
                    source_uri=f"media://{Path(wav_path).name}#t={start_ms / 1000:.3f},{end_ms / 1000:.3f}",
                    confidence=0.5,
                    attributes={
                        "backend": "energy-turn-detector",
                        "limitation": "turn index is not speaker identity",
                        "rms_threshold": threshold,
                    },
                )
            )
        return atoms
