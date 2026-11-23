"""Cheap, model-free signals that point at 'something happened here'.

Produces a per-second score in 0..1 from:
  * audio loudness spikes relative to the surrounding minute (shouting, laughing, hype)
  * YouTube's "most replayed" heatmap, when present

Twitch chat density and laughter/scream detection plug in here later.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class HotZone:
    t: float
    score: float
    reasons: list[str]


def loudness_per_second(wav_path: Path) -> np.ndarray:
    """RMS loudness in dBFS for each second of a 16-bit mono WAV, streamed in 1 s chunks."""
    out: list[float] = []
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        while True:
            frames = wf.readframes(rate)
            if not frames:
                break
            x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
            out.append(20.0 * np.log10(max(rms, 1e-6)))
    return np.asarray(out, dtype=np.float32)


def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    if x.size == 0:
        return x
    half = window // 2
    padded = np.pad(x, half, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    return np.median(view, axis=1)[: x.size]


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if x.size == 0 or window <= 1:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="same")


def loudness_score(db: np.ndarray) -> np.ndarray:
    """0..1: how much louder each second is than the minute around it (robust z-score)."""
    if db.size == 0:
        return db
    excess = db - _rolling_median(db, 61)
    mad = float(np.median(np.abs(excess - np.median(excess)))) * 1.4826
    z = excess / max(mad, 0.5)
    return np.clip(z / 4.0, 0.0, 1.0)


def heatmap_score(heatmap: list[dict[str, float]], seconds: int) -> np.ndarray:
    """YouTube 'most replayed' values spread onto a per-second array, contrast-stretched."""
    out = np.zeros(seconds, dtype=np.float32)
    if not heatmap or seconds == 0:
        return out
    for seg in heatmap:
        a = int(max(0, seg.get("start_time", 0)))
        b = int(min(seconds, np.ceil(seg.get("end_time", 0))))
        if b > a:
            out[a:b] = np.maximum(out[a:b], float(seg.get("value", 0)))
    # The graph's baseline is usually high; only the part above the median matters.
    base = float(np.median(out))
    top = float(out.max())
    if top <= base:
        return np.zeros_like(out)
    return np.clip((out - base) / (top - base), 0.0, 1.0)


@dataclass
class Signals:
    combined: np.ndarray            # per-second 0..1
    parts: dict[str, np.ndarray]    # name -> per-second 0..1

    def strength(self, start: float, end: float) -> float:
        a, b = int(max(0, start)), int(min(self.combined.size, np.ceil(end)))
        return float(self.combined[a:b].max()) if b > a else 0.0


def build_signals(duration: float, loudness_db: np.ndarray | None, heatmap: list[dict] | None) -> Signals:
    seconds = int(np.ceil(duration))
    parts: dict[str, np.ndarray] = {}
    if loudness_db is not None and loudness_db.size:
        loud = np.zeros(seconds, dtype=np.float32)
        n = min(seconds, loudness_db.size)
        loud[:n] = loudness_score(loudness_db[:n])
        parts["loudness spike"] = loud
    if heatmap:
        parts["replay peak"] = heatmap_score(heatmap, seconds)

    if not parts:
        return Signals(combined=np.zeros(seconds, dtype=np.float32), parts={})

    # Replay data is the stronger signal when it exists.
    weights = {"loudness spike": 1.0, "replay peak": 1.5}
    total = sum(weights.get(k, 1.0) for k in parts)
    combined = sum(_smooth(v, 5) * weights.get(k, 1.0) for k, v in parts.items()) / total
    peak = float(combined.max())
    if peak > 0:
        combined = combined / peak
    return Signals(combined=combined.astype(np.float32), parts=parts)


def hot_zones(signals: Signals, limit: int = 25, separation: int = 45, floor: float = 0.25) -> list[HotZone]:
    """Greedy peak picking: strongest second, suppress its neighbourhood, repeat."""
    score = signals.combined.copy()
    zones: list[HotZone] = []
    while len(zones) < limit and score.size:
        t = int(np.argmax(score))
        if score[t] < floor:
            break
        reasons = [name for name, arr in signals.parts.items() if arr[max(0, t - 3): t + 4].max() >= 0.5]
        zones.append(HotZone(t=float(t), score=float(signals.combined[t]), reasons=reasons))
        score[max(0, t - separation): t + separation + 1] = 0
    return zones
