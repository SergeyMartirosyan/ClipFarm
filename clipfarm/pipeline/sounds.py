"""What the stream sounds like, second by second: game action, crowd, and the streamer's reactions.

Loudness alone can't tell a laugh from an explosion, and a quiet streamer's big play still sounds
like one. YAMNet (Google's AudioSet classifier, 521 sound classes, a 4 MB TFLite model) labels
every second of audio; the classes are pooled into a few groups, and each group is judged against
its own level in this stream (a shooter is full of gunfire; only the bursts matter).

Runs on the CPU in well under a minute for a 2-hour stream. Optional: without the model file or
the runtime, the pipeline simply has one signal fewer.
"""

from __future__ import annotations

import csv
import logging
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW = 15600  # YAMNet input: 0.975 s

# YAMNet display names -> group. Matched exactly against yamnet_class_map.csv.
GROUPS: dict[str, tuple[str, ...]] = {
    "action": (
        "Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire", "Cap gun", "Explosion",
        "Boom", "Burst, pop", "Bang", "Smash, crash", "Breaking", "Shatter", "Glass", "Slam",
        "Skidding", "Tire squeal", "Car alarm", "Siren", "Police car (siren)", "Thunder",
    ),
    "crowd": ("Cheering", "Applause", "Clapping", "Crowd", "Roar"),
    "laughter": ("Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Baby laughter"),
    "scream": ("Screaming", "Shout", "Yell", "Whoop", "Children shouting", "Gasp", "Groan", "Wail, moan"),
}
# Readable names for the judge / vision prompts.
GROUP_WORDS = {"action": "game action", "crowd": "crowd", "laughter": "laughing", "scream": "screaming/shouting"}


@dataclass
class SoundData:
    seconds: int
    groups: dict[str, np.ndarray]                                # group -> per-second max class score
    top: list[str] = field(default_factory=list)                 # per second: strongest relevant class, or ""

    def summary(self, start: float, end: float, min_score: float = 0.3) -> str:
        """e.g. 'sounds: Explosion ×3, Screaming ×2' for the seconds in [start, end)."""
        a, b = int(max(0, start)), int(min(self.seconds, end))
        counts: dict[str, int] = {}
        for t in range(a, b):
            name = self.top[t] if t < len(self.top) else ""
            if name and max(arr[t] for arr in self.groups.values()) >= min_score:
                counts[name] = counts.get(name, 0) + 1
        if not counts:
            return ""
        best = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        return "sounds: " + ", ".join(f"{n} ×{c}" for n, c in best)


class SoundTagger:
    def __init__(self, model_path: Path, labels_path: Path, threads: int = 2):
        from ai_edge_litert.interpreter import Interpreter

        self.labels = [r["display_name"] for r in csv.DictReader(open(labels_path, encoding="utf-8"))]
        self.interpreter = Interpreter(model_path=str(model_path), num_threads=threads)
        self.interpreter.allocate_tensors()
        self._in = self.interpreter.get_input_details()[0]["index"]
        self._out = self.interpreter.get_output_details()[0]["index"]
        index = {name: i for i, name in enumerate(self.labels)}
        self.group_idx = {g: [index[n] for n in names if n in index] for g, names in GROUPS.items()}
        missing = [n for names in GROUPS.values() for n in names if n not in index]
        if missing:
            log.warning("Sound classes not in the label map (ignored): %s", missing)

    def scores(self, window: np.ndarray) -> np.ndarray:
        self.interpreter.set_tensor(self._in, window.astype(np.float32))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self._out)[0]

    def tag(self, wav_path: Path) -> SoundData:
        """One YAMNet window per second of a 16 kHz mono 16-bit WAV, streamed (low memory)."""
        groups: dict[str, list[float]] = {g: [] for g in self.group_idx}
        top: list[str] = []
        relevant = sorted({i for idx in self.group_idx.values() for i in idx})
        with wave.open(str(wav_path), "rb") as wf:
            if (wf.getframerate(), wf.getnchannels(), wf.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                raise ValueError("sound tagging needs 16 kHz mono 16-bit WAV")
            while True:
                frames = wf.readframes(SAMPLE_RATE)
                if not frames:
                    break
                x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                window = np.zeros(WINDOW, dtype=np.float32)
                n = min(WINDOW, x.size)
                window[:n] = x[:n]
                s = self.scores(window)
                for g, idx in self.group_idx.items():
                    groups[g].append(float(s[idx].max()) if idx else 0.0)
                best = max(relevant, key=lambda i: s[i]) if relevant else None
                top.append(self.labels[best] if best is not None and s[best] >= 0.2 else "")
        arrays = {g: np.asarray(v, dtype=np.float32) for g, v in groups.items()}
        return SoundData(seconds=len(top), groups=arrays, top=top)


def sound_signal(data: SoundData) -> np.ndarray:
    """0..1 per second: the strongest burst of any sound group above its own level in this stream.
    A group needs a real detection (score >= 0.15) to count at all."""
    from .signals import robust_burst

    if data.seconds == 0:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(data.seconds, dtype=np.float32)
    for arr in data.groups.values():
        if arr.size == 0 or float(arr.max()) < 0.15:
            continue
        burst = robust_burst(arr, window=121, smooth=3, min_scale=0.03)
        burst = np.where(arr >= 0.15, burst, 0.0)  # bursts of near-zero noise don't count
        out = np.maximum(out, burst[: data.seconds])
    return out


def load_tagger(model_path: Path, labels_path: Path) -> SoundTagger | None:
    if not Path(model_path).exists() or not Path(labels_path).exists():
        log.info("Sound model not found (%s); skipping game-sound signal. Run scripts/fetch_models.sh", model_path)
        return None
    try:
        return SoundTagger(model_path, labels_path)
    except ImportError:
        log.warning("ai-edge-litert not installed; skipping game-sound signal")
    except Exception as exc:
        log.warning("Sound model failed to load (%s); skipping game-sound signal", exc)
    return None
