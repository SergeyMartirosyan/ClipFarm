"""Turn the judge's rough proposals into clean, non-overlapping cut points."""

from __future__ import annotations

from dataclasses import replace

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
