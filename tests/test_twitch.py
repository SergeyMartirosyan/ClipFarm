"""Twitch signals, against mocked Helix / GQL responses (no network)."""

import json

import httpx
import numpy as np
import pytest

from clipfarm.pipeline import signals, twitch
from clipfarm.pipeline.judge import build_user_prompt


def test_vod_id_parsing():
    assert twitch.vod_id("https://www.twitch.tv/videos/2312345678") == "2312345678"
    assert twitch.vod_id("https://twitch.tv/somestreamer/video/123") == "123"
    assert twitch.vod_id("https://x.com/whatever", "v999", "twitchvod") == "999"
    assert twitch.vod_id("https://www.youtube.com/watch?v=abc", "abc", "youtube") is None


# --- Helix viewer clips ----------------------------------------------------------------------

def _helix_transport(pages):
    calls = {"clips": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "id.twitch.tv":
            return httpx.Response(200, json={"access_token": "tok"})
        assert request.headers["Authorization"] == "Bearer tok"
        if request.url.path.endswith("/videos"):
            return httpx.Response(200, json={"data": [{"id": "42", "user_id": "7", "created_at": "2026-11-01T18:00:00Z"}]})
        if request.url.path.endswith("/clips"):
            assert request.url.params["broadcaster_id"] == "7"
            page = pages[calls["clips"]]
            calls["clips"] += 1
            return httpx.Response(200, json=page)
        return httpx.Response(404)

    return httpx.MockTransport(handler), calls


def test_helix_vod_clips_paginates_and_filters():
    pages = [
        {"data": [
            {"video_id": "42", "vod_offset": 1200, "duration": 30, "view_count": 50, "title": "insane clutch", "url": "u1"},
            {"video_id": "99", "vod_offset": 10, "duration": 30, "view_count": 999, "title": "other vod", "url": "u2"},
            {"video_id": "42", "vod_offset": None, "duration": 30, "view_count": 5, "title": "no offset yet", "url": "u3"},
        ], "pagination": {"cursor": "next"}},
        {"data": [
            {"video_id": "42", "vod_offset": 300, "duration": 26.5, "view_count": 900, "title": "he fell off KEKW", "url": "u4"},
        ], "pagination": {}},
    ]
    transport, calls = _helix_transport(pages)
    helix = twitch.Helix("id", "secret", http=httpx.Client(transport=transport))
    clips = helix.vod_clips("42")
    assert calls["clips"] == 2
    assert [c.title for c in clips] == ["he fell off KEKW", "insane clutch"]  # most viewed first
    assert clips[0].offset == 300 and clips[0].end == pytest.approx(326.5)


def test_viewer_clip_signal_weights_popular_and_tail():
    clips = [twitch.ViewerClip(100, 30, 1000, "big", ""), twitch.ViewerClip(500, 30, 2, "small", "")]
    s = twitch.viewer_clip_signal(clips, 600)
    assert s.max() == pytest.approx(1.0)
    assert s[125] > s[105]          # tail of the clip (the moment) weighs more
    assert s[125] > s[525] > 0      # popular clip beats unpopular one
    assert s[50] == 0


# --- chat replay ------------------------------------------------------------------------------

def _gql_page(comments, has_next=True):
    return [{"data": {"video": {"comments": {
        "edges": [{"node": {"contentOffsetSeconds": t, "message": {"fragments": [{"text": txt}]}}} for t, txt in comments],
        "pageInfo": {"hasNextPage": has_next},
    }}}}]


def test_parse_comments_and_errors():
    msgs, more = twitch.parse_comments(_gql_page([(5, "hi"), (6, "KEKW")], False))
    assert msgs == [(5, "hi"), (6, "KEKW")] and more is False
    with pytest.raises(twitch.TwitchError):
        twitch.parse_comments([{"errors": [{"message": "PersistedQueryNotFound"}]}])
    with pytest.raises(twitch.TwitchError):
        twitch.parse_comments([{"data": {"video": None}}])


def test_classify_tokens():
    found = twitch.classify("KEKW KEKW what a fail, monkaS D: Pog")
    cats = [c for c, _ in found]
    assert cats.count("funny") == 2 and "shock" in cats and "hype" in cats


def test_chat_replay_walks_offsets_and_builds_burst():
    # Synthetic chat: 1 msg every 4 s, then a KEKW flood at 300-305 s.
    timeline = [(t, "chatting") for t in range(0, 600, 4)]
    timeline += [(t, "KEKW KEKW") for t in range(300, 306) for _ in range(8)]
    timeline.sort()
    requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body[0]["extensions"]["persistedQuery"]["sha256Hash"] == twitch.COMMENTS_QUERY_HASH
        offset = body[0]["variables"]["contentOffsetSeconds"]
        requested.append(offset)
        page = [m for m in timeline if m[0] >= offset][:20]  # Twitch returns small batches
        return httpx.Response(200, json=_gql_page(page, has_next=bool(page)))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    chat = twitch.ChatReplay(http=client, workers=2).fetch("42", 600)
    assert chat.total == len(timeline)  # every message counted exactly once
    assert chat.categories["funny"][300:306].sum() == 6 * 8 * 2

    sig = twitch.chat_signal(chat)
    peak = int(np.argmax(sig))
    assert 293 <= peak <= 302  # flood at 300 s, shifted ~5 s earlier to the moment
    assert "mostly funny" in chat.summary(300) and "KEKW" in chat.summary(300)


# --- combining with other signals ------------------------------------------------------------------

def test_audience_signals_outweigh_loudness():
    seconds = 600
    loud_db = np.full(seconds, -30.0, dtype=np.float32)
    loud_db[100:103] = -5.0  # a loud game explosion nobody cared about
    clips = [twitch.ViewerClip(400, 30, 500, "the real moment", "")]
    sig = signals.build_signals(
        seconds, loud_db, None, {signals.VIEWER_CLIPS: twitch.viewer_clip_signal(clips, seconds)}
    )
    assert sig.weights[signals.LOUDNESS] == signals.LOUDNESS_WEIGHT_WITH_AUDIENCE
    zones = signals.hot_zones(sig, limit=2)
    assert 410 <= zones[0].t <= 430
    assert signals.VIEWER_CLIPS in zones[0].reasons


def test_prompt_lists_viewer_clips_and_zone_notes():
    z = signals.HotZone(310, 0.9, ["chat burst"], notes=["chat 96 msgs/12s, mostly funny (KEKW ×90)"])
    p = build_user_prompt(
        title="t", uploader="u", duration=600, transcript="[1.0] hi", zones=[z], want=5,
        min_len=20, max_len=60, approved_examples=[], skipped_examples=[],
        viewer_clips=[twitch.ViewerClip(300, 30, 900, "he fell off KEKW", "")],
    )
    assert '300-330s  900 views  "he fell off KEKW"' in p
    assert "mostly funny (KEKW ×90)" in p
