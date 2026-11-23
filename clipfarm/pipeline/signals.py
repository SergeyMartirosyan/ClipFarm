"""Model-free signals that point at 'something happened here'.

Each signal is a per-second array in 0..1 (video time). Available ones:
  * loudness spike  - audio much louder than the surrounding minute
  * replay peak     - YouTube's "most replayed" heatmap
  * viewer clips    - Twitch clips viewers made from the VOD (pipeline/twitch.py)
  * chat burst      - Twitch chat activity + reaction emotes (pipeline/twitch.py)

They are combined with weights into one score used for hot zones and clip ranking.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LOUDNESS = "loudness spike"
REPLAY = "replay peak"
VIEWER_CLIPS = "viewer clips"
CHAT = "chat burst"

# How much each signal counts. Audience behaviour beats raw audio: loudness can't tell a
# laugh from an explosion, so it's down-weighted whenever an audience signal exists.
WEIGHTS = {VIEWER_CLIPS: 2.0, REPLAY: 1.5, CHAT: 1.5, LOUDNESS: 1.0}
LOUDNESS_WEIGHT_WITH_AUDIENCE = 0.5


@dataclass
class HotZone:
    t: float
    score: float
    reasons: list[str]
    notes: list[str] = field(default_factory=list)  # extra context for the judge


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


def robust_burst(x: np.ndarray, window: int = 61, smooth: int = 1, min_scale: float = 0.5) -> np.ndarray:
    """0..1: how far each second rises above its local median, in robust standard deviations
    (4 sigma = 1.0). Used for loudness and chat activity."""
    if x.size == 0:
        return x.astype(np.float32)
    x = _smooth(x.astype(np.float32), smooth)
    excess = x - _rolling_median(x, window)
    mad = float(np.median(np.abs(excess - np.median(excess)))) * 1.4826
    z = excess / max(mad, min_scale)
    return np.clip(z / 4.0, 0.0, 1.0).astype(np.float32)


def loudness_score(db: np.ndarray) -> np.ndarray:
    return robust_burst(db, window=61)


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
    weights: dict[str, float] = field(default_factory=dict)

    def strength(self, start: float, end: float) -> float:
        a, b = int(max(0, start)), int(min(self.combined.size, np.ceil(end)))
        return float(self.combined[a:b].max()) if b > a else 0.0

    def describe(self) -> str:
        return ", ".join(f"{k}×{self.weights.get(k, 1):g}" for k in self.parts) or "none"


def _fit(arr: np.ndarray, seconds: int) -> np.ndarray:
    out = np.zeros(seconds, dtype=np.float32)
    n = min(seconds, arr.size)
    out[:n] = arr[:n]
    return out


def build_signals(
    duration: float,
    loudness_db: np.ndarray | None = None,
    heatmap: list[dict] | None = None,
    extra: dict[str, np.ndarray] | None = None,
) -> Signals:
    seconds = int(np.ceil(duration))
    parts: dict[str, np.ndarray] = {}
    if loudness_db is not None and loudness_db.size:
        parts[LOUDNESS] = _fit(loudness_score(loudness_db), seconds)
    if heatmap:
        parts[REPLAY] = heatmap_score(heatmap, seconds)
    for name, arr in (extra or {}).items():
        if arr is not None and arr.size and float(arr.max()) > 0:
            parts[name] = _fit(arr, seconds)

    if not parts:
        return Signals(combined=np.zeros(seconds, dtype=np.float32), parts={})

    weights = {k: WEIGHTS.get(k, 1.0) for k in parts}
    if LOUDNESS in weights and len(parts) > 1:
        weights[LOUDNESS] = LOUDNESS_WEIGHT_WITH_AUDIENCE
    total = sum(weights.values())
    combined = sum(_smooth(v, 5) * weights[k] for k, v in parts.items()) / total
    peak = float(combined.max())
    if peak > 0:
        combined = combined / peak
    return Signals(combined=combined.astype(np.float32), parts=parts, weights=weights)


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
