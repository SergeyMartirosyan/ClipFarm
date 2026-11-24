"""Transcripts: YouTube json3 subtitles when available, otherwise faster-whisper."""

from __future__ import annotations

import json
import logging
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


def parse_json3(path: Path) -> list[Segment]:
    """YouTube json3 captions -> segments. Handles both manual and auto captions."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw: list[tuple[float, float, str]] = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs or ev.get("aAppend"):
            continue
        text = "".join(s.get("utf8", "") for s in segs).replace("\n", " ").strip()
        if not text:
            continue
        start = ev.get("tStartMs", 0) / 1000.0
        end = start + ev.get("dDurationMs", 0) / 1000.0
        raw.append((start, end, " ".join(text.split())))

    segments: list[Segment] = []
    for i, (start, end, text) in enumerate(raw):
        # Auto captions overlap (each line stays up until the next one ends); clamp.
        if i + 1 < len(raw):
            end = min(end, raw[i + 1][0]) if raw[i + 1][0] > start else end
        segments.append(Segment(start=start, end=max(end, start + 0.1), text=text))
    return segments


def load_wav(path: Path) -> np.ndarray:
    """16 kHz mono 16-bit WAV (what media.extract_audio writes) as float32 samples in -1..1.

    Whisper gets the samples directly instead of a file path: faster-whisper's own file decoding
    goes through PyAV, and some PyAV/faster-whisper version pairs are incompatible
    (av 19: "open() got an unexpected keyword argument 'metadata_errors'")."""
    with wave.open(str(path), "rb") as wf:
        if (wf.getframerate(), wf.getnchannels(), wf.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"{path} must be 16 kHz mono 16-bit WAV, got "
                             f"{wf.getframerate()} Hz, {wf.getnchannels()} ch, {8 * wf.getsampwidth()} bit")
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


class Transcriber:
    """Loads the Whisper model once per process, on first use."""

    def __init__(self, model: str = "small.en", compute_type: str = "int8", threads: int = 4):
        self.model_name = model
        self.compute_type = compute_type
        self.threads = threads
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            log.info("Loading Whisper model %s (%s)", self.model_name, self.compute_type)
            self._model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type=self.compute_type,
                cpu_threads=self.threads,
            )
        return self._model

    def transcribe(self, audio_path: Path, words: bool = False) -> list[Segment]:
        language = "en" if self.model_name.endswith(".en") else None
        audio = load_wav(audio_path) if Path(audio_path).suffix.lower() == ".wav" else str(audio_path)
        segments, _info = self.model.transcribe(
            audio,
            language=language,
            beam_size=1 if not words else 5,
            vad_filter=True,
            word_timestamps=words,
        )
        out: list[Segment] = []
        for s in segments:
            seg = Segment(start=float(s.start), end=float(s.end), text=s.text.strip())
            if words and s.words:
                seg.words = [
                    Word(start=float(w.start), end=float(w.end), text=w.word.strip())
                    for w in s.words
                    if w.word.strip()
                ]
            out.append(seg)
        return out


def save_segments(segments: list[Segment], path: Path) -> None:
    data = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)  # atomic: a crash never leaves a half-written transcript


def load_segments(path: Path) -> list[Segment]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return [Segment(float(d["start"]), float(d["end"]), str(d["text"])) for d in data]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def format_for_llm(segments: list[Segment]) -> str:
    """One line per segment: '[1234.5] text'. Compact, and timestamps are in seconds."""
    return "\n".join(f"[{s.start:.1f}] {s.text}" for s in segments if s.text)


def words_in_range(segments: list[Segment], start: float, end: float) -> list[Word]:
    return [w for s in segments for w in s.words if w.start >= start and w.end <= end]
