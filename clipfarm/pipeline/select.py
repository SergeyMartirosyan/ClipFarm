"""Turn the judge's rough proposals into clean, non-overlapping cut points."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .judge import Candidate
from .transcribe import Segment


def snap_boundaries(
    start: float,
    end: float,
    segments: list[Segment],
    cuts: list[float],
    min_len: float,
    max_len: float,
    duration: float,
) -> tuple[float, float]:
    """Move start/end onto sentence (segment) boundaries, then onto nearby shot cuts,
    then force the length into [min_len, max_len]."""
    # Sentence boundaries: start may move a bit earlier (keep the setup), end a bit later (keep the payoff).
    starts = [s.start for s in segments if start - 4.0 <= s.start <= start + 2.0]
    if starts:
        start = min(starts, key=lambda t: abs(t - start))
    ends = [s.end for s in segments if end - 2.0 <= s.end <= end + 4.0]
    if ends:
        end = min(ends, key=lambda t: abs(t - end))

    # A hard cut within 0.8 s of a boundary looks cleaner as the boundary itself.
    near_start = [c for c in cuts if abs(c - start) <= 0.8]
    if near_start:
        start = min(near_start, key=lambda c: abs(c - start))
    near_end = [c for c in cuts if abs(c - end) <= 0.8 and c > start]
    if near_end:
        end = min(near_end, key=lambda c: abs(c - end))

    # Length limits.
    if end - start > max_len:
        fitting = [s.end for s in segments if start + min_len <= s.end <= start + max_len]
        end = max(fitting) if fitting else start + max_len
    if end - start < min_len:
        longer = [s.end for s in segments if start + min_len <= s.end <= start + max_len]
        end = min(longer) if longer else start + min_len

    start = max(0.0, start)
    end = min(duration, end)
    if end - start < min_len and duration >= min_len:  # hit the end of the video: pull start back
        start = max(0.0, end - min_len)
    return round(start, 2), round(end, 2)


def overlaps(a: tuple[float, float], b: tuple[float, float], tolerance: float = 3.0) -> bool:
    return a[0] < b[1] - tolerance and b[0] < a[1] - tolerance


def choose(candidates: list[Candidate], limit: int) -> list[Candidate]:
    """Best final score first, dropping anything that overlaps an already-chosen clip."""
    chosen: list[Candidate] = []
    for c in sorted(candidates, key=lambda c: c.final_score, reverse=True):
        if any(overlaps((c.start, c.end), (k.start, k.end)) for k in chosen):
            continue
        chosen.append(c)
        if len(chosen) >= limit:
            break
    return chosen


def with_bounds(c: Candidate, start: float, end: float) -> Candidate:
    return replace(c, start=start, end=end)


LEAD_IN = 0.3  # seconds kept before the first word


def dead_air_trim(lead: float, start: float, length: float, min_len: float,
                  loud_db: np.ndarray | None = None, sound: np.ndarray | None = None) -> float:
    """Seconds to cut from the start of a clip whose first word comes `lead` seconds in.

    Short-form viewers decide in the first 2 seconds, so silence there costs views. But in gaming
    the stretch before anyone talks can be the action itself, so only *dead* air is cut: the
    lead-in must have no game sounds or reactions and be no louder than the stream's quiet
    level. Keeps LEAD_IN seconds before the first word and never goes below `min_len`."""
    if lead < 1.0:
        return 0.0
    cut = min(lead - LEAD_IN, max(0.0, length - min_len))
    if cut < 0.7:
        return 0.0
    a, b = int(max(0, start)), int(math.ceil(start + lead))
    if sound is not None and sound[a:b].size and float(sound[a:b].max()) >= 0.3:
        return 0.0
    if loud_db is not None and loud_db.size >= b > a:
        if float(loud_db[a:b].mean()) > float(np.percentile(loud_db, 40)):
            return 0.0  # loud lead-in: probably action, keep it
    return round(cut, 2)
