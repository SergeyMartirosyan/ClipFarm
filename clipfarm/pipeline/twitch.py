"""Twitch signals for VODs: what viewers clipped, and how chat reacted.

Viewer clips come from the official Helix API (needs a free Twitch app: client id + secret).
Chat replay comes from Twitch's public web GraphQL endpoint (the same one the website and
tools like TwitchDownloader use). It is unofficial: if Twitch changes it, we log a warning
and carry on without chat.
"""

from __future__ import annotations

import bisect
import logging
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
import numpy as np

log = logging.getLogger(__name__)

HELIX = "https://api.twitch.tv/helix"
TOKEN_URL = "https://id.twitch.tv/oauth2/token"
GQL_URL = "https://gql.twitch.tv/gql"
# Twitch's own public web client id; chat replay is readable without logging in.
GQL_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"
COMMENTS_QUERY_HASH = "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a"

# Chat reacts after the moment (stream delay + typing time).
CHAT_DELAY_SECONDS = 5

# Lower-cased tokens. Includes common BTTV/7TV/FFZ emotes, which arrive as plain text.
EMOTES = {
    "funny": {
        "kekw", "lul", "lulw", "omegalul", "icant", "kekl", "lmao", "lmfao", "lol", "xd", "xdd",
        "pepelaugh", "kappa", "4head", "😂", "🤣", "haha", "hahaha", "dead", "💀", "clueless",
    },
    "hype": {
        "pog", "pogchamp", "poggers", "pogu", "w", "letsgo", "lets", "ez", "clap", "gigachad",
        "goated", "insane", "huge", "peepoclap", "catjam", "🔥", "pogcrazy", "omg", "holy",
    },
    "shock": {
        "monkas", "monkaw", "d:", "wutface", "wtf", "notlikethis", "f", "sadge", "aware", "nooo",
        "noooo", "widepeeposad", "pepehands", "😱", "rip", "o7", "modcheck",
    },
}
_TOKEN_TO_CATEGORY = {tok: cat for cat, toks in EMOTES.items() for tok in toks}
_VOD_RE = re.compile(r"twitch\.tv/(?:[^/]+/)?videos?/(\d+)")


class TwitchError(RuntimeError):
    pass


def vod_id(url: str, info_id: str = "", extractor: str = "") -> str | None:
    m = _VOD_RE.search(url or "")
    if m:
        return m.group(1)
    if "twitch" in (extractor or "") and "vod" in (extractor or "") and info_id:
        return info_id.lstrip("v")
    return None


# ---- viewer clips (Helix) ------------------------------------------------------------------


@dataclass
class ViewerClip:
    offset: float      # seconds into the VOD where the clip starts
    duration: float
    views: int
    title: str
    url: str

    @property
    def end(self) -> float:
        return self.offset + self.duration


class Helix:
    def __init__(self, client_id: str, client_secret: str, http: httpx.Client | None = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.http = http or httpx.Client(timeout=20)
        self._token: str | None = None

    def _auth(self) -> dict:
        if self._token is None:
            r = self.http.post(
                TOKEN_URL,
                params={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "grant_type": "client_credentials",
                },
            )
            if r.status_code != 200:
                raise TwitchError(f"Twitch auth failed ({r.status_code}): check TWITCH_CLIENT_ID/SECRET")
            self._token = r.json()["access_token"]
        return {"Client-Id": self.client_id, "Authorization": f"Bearer {self._token}"}

    def _get(self, path: str, params: dict) -> dict:
        r = self.http.get(f"{HELIX}/{path}", params=params, headers=self._auth())
        if r.status_code != 200:
            raise TwitchError(f"Helix {path} failed ({r.status_code}): {r.text[:200]}")
        return r.json()

    def video(self, vid: str) -> dict:
        data = self._get("videos", {"id": vid}).get("data") or []
        if not data:
            raise TwitchError(f"Twitch VOD {vid} not found")
        return data[0]

    def vod_clips(self, vid: str, max_pages: int = 10) -> list[ViewerClip]:
        """All clips viewers made from this VOD (live or afterwards), most viewed first."""
        video = self.video(vid)
        params = {
            "broadcaster_id": video["user_id"],
            "started_at": video["created_at"],
            "ended_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "first": 100,
        }
        clips: list[ViewerClip] = []
        for _ in range(max_pages):
            page = self._get("clips", params)
            clips += parse_helix_clips(page, vid)
            cursor = (page.get("pagination") or {}).get("cursor")
            if not cursor:
                break
            params["after"] = cursor
        clips.sort(key=lambda c: c.views, reverse=True)
        return clips


def parse_helix_clips(page: dict, vid: str) -> list[ViewerClip]:
    out = []
    for c in page.get("data") or []:
        if str(c.get("video_id")) != str(vid) or c.get("vod_offset") is None:
            continue
        out.append(
            ViewerClip(
                offset=float(c["vod_offset"]),
                duration=float(c.get("duration") or 30),
                views=int(c.get("view_count") or 0),
                title=(c.get("title") or "").strip(),
                url=c.get("url") or "",
            )
        )
    return out


def viewer_clip_signal(clips: list[ViewerClip], seconds: int) -> np.ndarray:
    """0..1 per second. Popular clips weigh more; the last 40% of each clip (where the
    moment usually is: people press 'clip' right after it) weighs double."""
    out = np.zeros(seconds, dtype=np.float32)
    for c in clips:
        a, b = int(max(0, c.offset)), int(min(seconds, math.ceil(c.end)))
        if b <= a:
            continue
        w = 1.0 + math.log1p(c.views)
        out[a:b] += w * 0.5
        tail = a + int((b - a) * 0.6)
        out[tail:b] += w * 0.5
    peak = float(out.max()) if out.size else 0.0
    return out / peak if peak > 0 else out


def clips_near(clips: list[ViewerClip], start: float, end: float) -> list[ViewerClip]:
    return [c for c in clips if c.offset < end and c.end > start]


# ---- chat replay (GQL) -------------------------------------------------------------------------


@dataclass
class ChatData:
    seconds: int
    rate: np.ndarray                                   # messages per second
    categories: dict[str, np.ndarray]                  # category -> reaction tokens per second
    emote_events: list[tuple[int, str]] = field(default_factory=list)  # (second, token), sorted

    @property
    def total(self) -> int:
        return int(self.rate.sum())

    def summary(self, t: float, window: int = 12) -> str:
        """Human-readable reaction around second t (chat time, i.e. after the moment)."""
        a = int(max(0, t))
        b = min(self.seconds, a + window)
        msgs = int(self.rate[a:b].sum())
        if msgs == 0:
            return ""
        cats = {k: int(v[a:b].sum()) for k, v in self.categories.items()}
        top_cat = max(cats, key=cats.get) if any(cats.values()) else ""
        times = [e[0] for e in self.emote_events]
        lo, hi = bisect.bisect_left(times, a), bisect.bisect_left(times, b)
        top = Counter(tok for _, tok in self.emote_events[lo:hi]).most_common(2)
        tokens = ", ".join(f"{tok} ×{n}" for tok, n in top)
        mood = f"mostly {top_cat}" if top_cat and cats[top_cat] >= 3 else "mixed"
        return f"chat {msgs} msgs/{b - a}s, {mood}" + (f" ({tokens})" if tokens else "")


def classify(text: str) -> list[tuple[str, str]]:
    """(category, token) for each reaction token in a message."""
    found = []
    for raw in text.split():
        tok = raw.strip(".,!?:;\"'()").lower()
        if raw.strip() == "D:":
            tok = "d:"
        cat = _TOKEN_TO_CATEGORY.get(tok)
        if cat:
            found.append((cat, raw.strip(".,!?\"'()")))
    return found


def parse_comments(payload) -> tuple[list[tuple[int, str]], bool]:
    """GQL response -> ([(offset_seconds, text)], has_next_page)."""
    item = payload[0] if isinstance(payload, list) else payload
    if item.get("errors"):
        raise TwitchError(f"Twitch chat API error: {item['errors'][0].get('message')}")
    video = (item.get("data") or {}).get("video")
    if video is None:
        raise TwitchError("Twitch chat API returned no video (VOD private, deleted, or API changed)")
    comments = video.get("comments") or {}
    out = []
    for edge in comments.get("edges") or []:
        node = edge.get("node") or {}
        frags = (node.get("message") or {}).get("fragments") or []
        text = "".join(f.get("text") or "" for f in frags)
        out.append((int(node.get("contentOffsetSeconds") or 0), text))
    return out, bool((comments.get("pageInfo") or {}).get("hasNextPage"))


class ChatReplay:
    def __init__(self, http: httpx.Client | None = None, workers: int = 4, max_requests: int = 8000):
        self.http = http or httpx.Client(timeout=20, headers={"Client-Id": GQL_CLIENT_ID})
        self.workers = workers
        self.max_requests = max_requests

    def _page(self, vid: str, offset: int):
        body = [{
            "operationName": "VideoCommentsByOffsetOrCursor",
            "variables": {"videoID": vid, "contentOffsetSeconds": offset},
            "extensions": {"persistedQuery": {"version": 1, "sha256Hash": COMMENTS_QUERY_HASH}},
        }]
        r = self.http.post(GQL_URL, json=body, headers={"Client-Id": GQL_CLIENT_ID})
        if r.status_code != 200:
            raise TwitchError(f"Twitch chat API HTTP {r.status_code}")
        return parse_comments(r.json())

    def _range(self, vid: str, start: int, end: int, budget: int) -> list[tuple[int, str]]:
        """Walk the chat by offset (cursor paging needs an integrity token; offsets don't)."""
        msgs: list[tuple[int, str]] = []
        offset = start
        for _ in range(budget):
            if offset >= end:
                break
            page, has_next = self._page(vid, offset)
            page = [m for m in page if m[0] >= offset]
            if not page:
                if not has_next:
                    break
                offset += 10  # quiet stretch
                continue
            last = max(t for t, _ in page)
            if last > offset:
                # The batch may have cut the last second short: keep everything before it and
                # re-read that second from its start on the next request.
                msgs += [m for m in page if m[0] < last and m[0] < end]
                offset = last
            else:
                # A whole batch inside one second (a huge spam burst): take it and move on.
                msgs += [m for m in page if m[0] < end]
                offset += 1
        return msgs

    def fetch(self, vid: str, duration: float) -> ChatData:
        seconds = int(math.ceil(duration))
        n = max(1, self.workers)
        bounds = [int(seconds * i / n) for i in range(n + 1)]
        budget = self.max_requests // n
        with ThreadPoolExecutor(n) as pool:
            parts = list(pool.map(lambda i: self._range(vid, bounds[i], bounds[i + 1], budget), range(n)))
        return build_chat_data([m for p in parts for m in p], seconds)


def build_chat_data(messages: list[tuple[int, str]], seconds: int) -> ChatData:
    rate = np.zeros(seconds, dtype=np.float32)
    cats = {c: np.zeros(seconds, dtype=np.float32) for c in EMOTES}
    events: list[tuple[int, str]] = []
    for t, text in messages:
        if not 0 <= t < seconds:
            continue
        rate[t] += 1
        for cat, tok in classify(text):
            cats[cat][t] += 1
            events.append((t, tok))
    events.sort(key=lambda e: e[0])
    return ChatData(seconds=seconds, rate=rate, categories=cats, emote_events=events)


def chat_signal(chat: ChatData) -> np.ndarray:
    """0..1 per second, in *video* time: bursts vs. the surrounding two minutes,
    reaction emotes counting extra, shifted earlier by the chat delay."""
    from .signals import robust_burst  # local import: signals imports nothing from here

    reactions = sum(chat.categories.values())
    activity = chat.rate + 0.5 * reactions
    burst = robust_burst(activity, window=121, smooth=5)
    shifted = np.zeros_like(burst)
    d = CHAT_DELAY_SECONDS
    if burst.size > d:
        shifted[: burst.size - d] = burst[d:]
    return shifted
