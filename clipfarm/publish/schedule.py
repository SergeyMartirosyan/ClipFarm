"""Posting slots: spread approved clips over the day instead of posting them in a burst.

Each platform gets `per_day` evenly spaced slots inside a daily window (e.g. 10:00-23:00 local
time). An approved clip takes the earliest free slot that is at least a couple of minutes away;
a few minutes of jitter keeps the rhythm from looking robotic.
"""

from __future__ import annotations

import random
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

MIN_LEAD = 120  # seconds: never schedule a post sooner than this
SLOT_MATCH = 60  # an existing post within this many seconds of a slot occupies it


def parse_window(window: str) -> tuple[time, time]:
    start, end = (part.strip() for part in window.split("-", 1))
    t0 = time.fromisoformat(start)
    t1 = time.fromisoformat(end)
    if t1 <= t0:
        raise ValueError(f"POST_WINDOW must be like 10:00-23:00, got {window!r}")
    return t0, t1


def day_slots(day: datetime, per_day: int, window: tuple[time, time], tz: ZoneInfo) -> list[float]:
    """Unix times of the `per_day` slots on the local calendar day of `day`."""
    t0, t1 = window
    start = datetime.combine(day.date(), t0, tzinfo=tz)
    end = datetime.combine(day.date(), t1, tzinfo=tz)
    step = (end - start) / per_day
    # Centre each slot in its share of the window.
    return [(start + step * (k + 0.5)).timestamp() for k in range(per_day)]


def next_slot(now: float, taken: list[float], per_day: int, window: str, tz_name: str,
              jitter: float = 240.0, rng: random.Random | None = None, horizon_days: int = 60) -> float:
    """Earliest free slot at least MIN_LEAD seconds from now. Returns a unix time."""
    rng = rng or random.Random()
    tz = ZoneInfo(tz_name)
    win = parse_window(window)
    local_now = datetime.fromtimestamp(now, tz)
    for d in range(horizon_days):
        day = local_now + timedelta(days=d)
        for slot in day_slots(day, max(1, per_day), win, tz):
            if slot < now + MIN_LEAD:
                continue
            if any(abs(slot - t) <= SLOT_MATCH + jitter for t in taken):
                continue
            return slot + rng.uniform(-jitter, jitter) if jitter else slot
    raise RuntimeError("No free posting slot in the next 60 days; raise POSTS_PER_DAY")


def describe(ts: float, tz_name: str, now: float | None = None) -> str:
    """'today 18:40', 'tomorrow 10:25', 'Thu 12:05' in the posting time zone."""
    tz = ZoneInfo(tz_name)
    when = datetime.fromtimestamp(ts, tz)
    today = datetime.fromtimestamp(now, tz).date() if now is not None else datetime.now(tz).date()
    if when.date() == today:
        day = "today"
    elif when.date() == today + timedelta(days=1):
        day = "tomorrow"
    else:
        day = when.strftime("%a %d %b")
    return f"{day} {when:%H:%M}"
