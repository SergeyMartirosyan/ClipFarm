"""Picking upgrades: game-sound tagging, chat replay robustness, public viewer clips, dead-air trim."""

import json
import shutil
import subprocess
from pathlib import Path

import httpx
import numpy as np
import pytest

from clipfarm.pipeline import select, signals, sounds, twitch

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "data" / "models" / "yamnet.tflite"
LABELS = REPO / "data" / "models" / "yamnet_class_map.csv"
needs_model = pytest.mark.skipif(
    not (MODEL.exists() and LABELS.exists()) or shutil.which("ffmpeg") is None,
    reason="YAMNet model not downloaded (scripts/fetch_models.sh) or ffmpeg missing",
)


# --- game sounds --------------------------------------------------------------------------------

@needs_model
def test_real_yamnet_finds_explosion_at_the_right_second(tmp_path):
    wav = tmp_path / "boom.wav"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "anoisesrc=d=60:c=brown:a=0.01",
        "-f", "lavfi", "-i", "anoisesrc=d=60:c=brown:a=1",
        "-filter_complex", "[1]lowpass=f=400,volume='if(between(t,30,33),exp(-(t-30)*1.5),0)':eval=frame[b];"
                           "[0][b]amix=inputs=2:normalize=0,aresample=16000",
        "-ac", "1", "-c:a", "pcm_s16le", str(wav)], check=True)
    tagger = sounds.load_tagger(MODEL, LABELS)
    data = tagger.tag(wav)
    assert data.seconds == 60
    assert data.groups["action"][30] > 0.3 and data.groups["action"][:25].max() < 0.1
    sig = sounds.sound_signal(data)
    assert int(np.argmax(sig)) in (29, 30, 31)
    assert "Explosion" in data.summary(25, 40)


def test_sound_signal_ignores_constant_background_and_noise():
    n = 300
    groups = {g: np.zeros(n, dtype=np.float32) for g in sounds.GROUPS}
    groups["action"][:] = 0.5            # a shooter: gunfire all the time
    groups["action"][150:153] = 0.95     # one huge moment
    groups["laughter"][:] = 0.05         # near-zero noise never counts
    groups["laughter"][200] = 0.12
    data = sounds.SoundData(seconds=n, groups=groups, top=[""] * n)
    sig = sounds.sound_signal(data)
    assert 149 <= int(np.argmax(sig)) <= 153
    assert sig[200] == 0 and sig[50] < 0.2


def test_missing_model_is_skipped(tmp_path):
    assert sounds.load_tagger(tmp_path / "nope.tflite", tmp_path / "nope.csv") is None


def test_sound_notes_reach_the_judge():
    from clipfarm.pipeline.run import annotate_zones, audience_note

    n = 100
    groups = {g: np.zeros(n, dtype=np.float32) for g in sounds.GROUPS}
    groups["scream"][40:43] = 0.8
    top = [""] * n
    top[40:43] = ["Screaming"] * 3
    data = sounds.SoundData(seconds=n, groups=groups, top=top)
    zone = signals.HotZone(41, 0.9, ["game/reaction sound"])
    annotate_zones([zone], None, data)
    assert zone.notes == ["sounds: Screaming ×3"]
    assert audience_note(None, 35, 60, data) == "sounds: Screaming ×3"


# --- chat replay robustness -----------------------------------------------------------------------

def _gql(comments, has_next=True):
    return [{"data": {"video": {"comments": {
        "edges": [{"node": {"contentOffsetSeconds": t, "message": {"fragments": [{"text": x}]}}} for t, x in comments],
        "pageInfo": {"hasNextPage": has_next}}}}}]


def _chat_client(timeline, fail):
    def handler(req):
        offset = json.loads(req.content)[0]["variables"]["contentOffsetSeconds"]
        verdict = fail(offset)
        if verdict:
            return httpx.Response(verdict)
        page = [m for m in timeline if m[0] >= offset][:20]
        return httpx.Response(200, json=_gql(page, bool(page)))
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_chat_page_errors_are_retried():
    timeline = [(t, "hi") for t in range(0, 400, 2)]
    flaky = {"n": 0}

    def fail(offset):
        flaky["n"] += 1
        return 502 if flaky["n"] % 3 == 1 else None  # every 3rd request fails once

    chat = twitch.ChatReplay(http=_chat_client(timeline, fail), workers=2, sleep=lambda s: None).fetch("1", 400)
    assert chat.total == len(timeline) and chat.coverage == 1.0 and not chat.problems


def test_chat_keeps_what_it_read_when_a_range_dies():
    timeline = [(t, "KEKW") for t in range(0, 400, 2)]
    chat = twitch.ChatReplay(http=_chat_client(timeline, lambda o: 500 if o >= 300 else None),
                             workers=2, sleep=lambda s: None).fetch("1", 400)
    assert 0.7 <= chat.coverage < 1.0
    assert chat.total >= 140 and chat.problems  # first range complete, second partial
    assert twitch.chat_signal(chat).shape == (400,)


def test_chat_total_failure_still_raises():
    with pytest.raises(twitch.TwitchError):
        twitch.ChatReplay(http=_chat_client([], lambda o: 503), workers=2, sleep=lambda s: None).fetch("1", 100)


def test_chat_budget_is_shared_and_reported():
    timeline = [(t, "x") for t in range(0, 400)]
    chat = twitch.ChatReplay(http=_chat_client(timeline, lambda o: None), workers=2, max_requests=6,
                             sleep=lambda s: None).fetch("1", 400)
    assert chat.coverage < 1.0 and "budget" in " ".join(chat.problems)


# --- viewer clips via the public route -----------------------------------------------------------

def test_clips_period_by_vod_age():
    now = 1_000_000_000
    assert twitch.clips_period(now - 3600, now) == "LAST_DAY"
    assert twitch.clips_period(now - 3 * 86400, now) == "LAST_WEEK"
    assert twitch.clips_period(now - 20 * 86400, now) == "LAST_MONTH"
    assert twitch.clips_period(now - 90 * 86400, now) == "ALL_TIME"
    assert twitch.clips_period(0, now) == "LAST_MONTH"


def test_public_clips_query_pages_and_filters_to_the_vod():
    pages = [
        {"data": {"user": {"clips": {"pageInfo": {"hasNextPage": True}, "edges": [
            {"cursor": "c1", "node": {"id": "AbC", "title": "speed falls KEKW", "viewCount": 900, "durationSeconds": 28,
                                      "videoOffsetSeconds": 1290, "video": {"id": "42"}}},
            {"cursor": "c2", "node": {"id": "X", "title": "other stream", "viewCount": 5000, "durationSeconds": 30,
                                      "videoOffsetSeconds": 10, "video": {"id": "7"}}},
        ]}}}},
        {"data": {"user": {"clips": {"pageInfo": {"hasNextPage": False}, "edges": [
            {"cursor": "c3", "node": {"id": "Y", "title": "clutch", "viewCount": 50, "durationSeconds": 30,
                                      "videoOffsetSeconds": 3500, "video": {"id": "42"}}},
            {"cursor": "c4", "node": {"id": "Z", "title": "vod deleted", "viewCount": 9, "durationSeconds": 30,
                                      "videoOffsetSeconds": None, "video": {"id": "42"}}},
        ]}}}},
    ]
    queries = []

    def handler(req):
        q = json.loads(req.content)["query"]
        queries.append(q)
        assert req.headers["Client-Id"] == twitch.GQL_CLIENT_ID
        return httpx.Response(200, json=pages[len(queries) - 1])

    clips = twitch.public_vod_clips("ishowspeed", "42", 0, http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert [c.title for c in clips] == ["speed falls KEKW", "clutch"]
    assert clips[0].url == "https://clips.twitch.tv/AbC" and clips[0].offset == 1290
    assert 'user(login: "ishowspeed")' in queries[0] and "after" not in queries[0]
    assert 'after: "c2"' in queries[1] and "period: LAST_MONTH" in queries[1]


def test_public_clips_errors_raise():
    def handler(req):
        return httpx.Response(200, json={"errors": [{"message": "service error"}]})

    with pytest.raises(twitch.TwitchError):
        twitch.public_vod_clips("x", "1", http=httpx.Client(transport=httpx.MockTransport(handler)))


# --- dead air at the start of a clip -------------------------------------------------------------

def test_dead_air_is_trimmed_only_when_quiet():
    loud = np.full(600, -20.0, dtype=np.float32)
    loud[100:104] = -45.0                                   # quiet lead-in at 100 s
    assert select.dead_air_trim(3.0, 100, 40, 15, loud_db=loud) == pytest.approx(2.7)
    assert select.dead_air_trim(0.8, 100, 40, 15, loud_db=loud) == 0      # short lead: keep
    loud_action = np.full(600, -30.0, dtype=np.float32)
    loud_action[200:204] = -8.0                             # loud lead-in: probably action
    assert select.dead_air_trim(3.0, 200, 40, 15, loud_db=loud_action) == 0
    sound = np.zeros(600, dtype=np.float32)
    sound[101] = 0.9                                        # an explosion before the first word
    assert select.dead_air_trim(3.0, 100, 40, 15, loud_db=loud, sound=sound) == 0
    assert select.dead_air_trim(3.0, 100, 16, 15, loud_db=loud) == pytest.approx(1.0)  # capped by min length
    assert select.dead_air_trim(3.0, 100, 15.5, 15, loud_db=loud) == 0  # too little room to bother
