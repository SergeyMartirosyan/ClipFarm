"""Long streams: time ranges in messages, splitting into parts, and scouting the hottest hours.

A 2-hour 1080p60 VOD is ~7 GB and needs ~15 GB of disk while downloading; Whisper takes ~45 min on
two cores. So long streams are never processed in one piece:

  * a range in the message ("<link> 1:30:00-3:30:00", or a Twitch ?t= timestamp) processes only
    that part;
  * streams up to SCOUT_ABOVE_HOURS are split into PART_HOURS parts, queued one after another;
  * longer ones are scouted first: chat activity and viewer clips over the whole VOD (cheap, no
    video download) pick the hottest SCOUT_PARTS windows, and only those are processed.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

import numpy as np

OVERLAP = 60  # seconds shared by neighbouring parts, so a moment on the boundary isn't cut in two

_CLOCK = r"\d{1,3}:\d{2}(?::\d{2})?"
_UNITS = r"(?:\d+h)?(?:\d+m)?(?:\d+s)?"
_TIME_RE = re.compile(rf"(?<![\w:])({_CLOCK}|\d+h(?:\d+m)?(?:\d+s)?|\d+m(?:\d+s)?|\d+s)(?![\w:])", re.I)


def parse_time(token: str) -> float | None:
    """'1:30:00', '90:00', '1h30m', '45m', '20s' -> seconds."""
    token = token.strip().lower()
    if ":" in token:
        parts = [int(p) for p in token.split(":")]
        if len(parts) == 2:
            return float(parts[0] * 60 + parts[1])
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", token)
    if not m or not any(m.groups()):
        return None
    h, mi, s = (int(x) if x else 0 for x in m.groups())
    return float(h * 3600 + mi * 60 + s)


def url_timestamp(url: str) -> float | None:
    """Twitch/YouTube '?t=1h2m3s' or '?t=3723'."""
    values = parse_qs(urlparse(url).query).get("t")
    if not values:
        return None
    v = values[0]
    return float(v) if v.isdigit() else parse_time(v)


def parse_range(text_after_url: str, url: str = "", default_len: float = 7200) -> tuple[float, float] | None:
    """A time range written after the link ("1:30:00-3:30:00", "from 1h30m to 3h", "2h"), else the
    link's own timestamp. A single time means: from there, `default_len` seconds."""
    times = [t for t in (parse_time(m) for m in _TIME_RE.findall(text_after_url)) if t is not None]
    if len(times) >= 2 and times[0] != times[1]:
        return min(times[:2]), max(times[:2])  # "3:00:00-1:00:00" is a typo for the other way round
    if len(times) == 1:
        return times[0], times[0] + default_len
    ts = url_timestamp(url)
    if ts is not None:
        return ts, ts + default_len
    return None


def hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def split_ranges(start: float, end: float, part: float, overlap: float = OVERLAP) -> list[tuple[float, float]]:
    """Consecutive parts of at most `part` seconds (+ overlap). A short leftover joins the last part."""
    if end - start <= part * 1.25:
        return [(start, end)]
    out: list[tuple[float, float]] = []
    t = start
    while t < end:
        stop = min(end, t + part)
        if end - stop < part * 0.25:  # tiny tail: absorb it
            stop = end
        out.append((t, min(end, stop + overlap) if stop < end else end))
        t = stop
    return out


def pick_hot_windows(score: np.ndarray, window: int, count: int) -> list[tuple[int, int]]:
    """Best `count` non-overlapping windows of `window` units (e.g. minutes) by summed score."""
    if score.size == 0 or window <= 0:
        return []
    window = min(window, score.size)
    sums = np.convolve(score, np.ones(window), mode="valid")  # sums[i] = score[i:i+window].sum()
    taken: list[tuple[int, int]] = []
    sums = sums.astype(float).copy()
    for _ in range(count):
        if sums.size == 0 or float(sums.max()) <= 0:
            break
        i = int(np.argmax(sums))
        taken.append((i, i + window))
        lo, hi = max(0, i - window + 1), min(sums.size, i + window)
        sums[lo:hi] = -1  # no overlap with this window
    return sorted(taken)


def activity_per_minute(minutes: int, chat_rate: np.ndarray | None = None, clip_offsets=(),
                        heatmap: list[dict] | None = None) -> np.ndarray:
    """0..~3 per minute from whatever is available: chat messages/s (relative to the stream's own
    level), viewer clips (weighted by views) and the YouTube replay heatmap."""
    score = np.zeros(minutes, dtype=np.float32)
    if chat_rate is not None and chat_rate.size:
        r = chat_rate[:minutes].astype(np.float32)
        ref = float(np.percentile(r[r > 0], 90)) if np.any(r > 0) else 0.0
        if ref > 0:
            score[: r.size] += np.clip(r / ref, 0, 2)
    for offset, views in clip_offsets:
        m = int(offset // 60)
        if 0 <= m < minutes:
            score[m] += 0.5 + np.log1p(views) / 4
    for seg in heatmap or []:
        a, b = int(seg.get("start_time", 0) // 60), int(seg.get("end_time", 0) // 60) + 1
        score[max(0, a):min(minutes, b)] += float(seg.get("value", 0))
    return score


def describe_parts(ranges: list[tuple[float, float]]) -> str:
    return ", ".join(f"{hms(a)}–{hms(b)}" for a, b in ranges)
