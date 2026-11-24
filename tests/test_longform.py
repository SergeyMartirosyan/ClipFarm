"""Long streams: ranges in messages, splitting into parts, scouting, and section offsets."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from clipfarm import db as dbm
from clipfarm import longform


# --- parsing --------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, url, expected", [
    (" 1:30:00-3:30:00", "", (5400, 12600)),
    (" from 1h30m to 3h please", "", (5400, 10800)),
    (" 90:00 - 120:00", "", (5400, 7200)),
    (" 45m", "", (2700, 2700 + 7200)),
    ("", "https://www.twitch.tv/videos/1?t=1h2m3s", (3723, 3723 + 7200)),
    ("", "https://youtu.be/x?t=600", (600, 7800)),
    (" what a stream!", "https://www.twitch.tv/videos/1", None),
    (" 3:00:00-1:00:00", "", (3600, 10800)),  # backwards: swapped
])
def test_parse_range(text, url, expected):
    got = longform.parse_range(text, url)
    assert got == (pytest.approx(expected[0]), pytest.approx(expected[1])) if expected else got is None


def test_split_ranges_overlap_and_short_tail():
    parts = longform.split_ranges(0, 10 * 3600, 7200)
    assert len(parts) == 5 and parts[0] == (0, 7260) and parts[-1] == (28800, 36000)
    assert longform.split_ranges(0, 2.4 * 3600, 7200) == [(0, 2.4 * 3600)]           # short enough: one piece
    two = longform.split_ranges(0, 4.3 * 3600, 7200)
    assert len(two) == 2 and two[-1][1] == 4.3 * 3600                                   # tail joined to the last part


def test_hot_windows_and_activity():
    minutes = 44 * 60
    chat = np.full(minutes, 2.0, dtype=np.float32)
    chat[1500:1560] = 20.0                                   # a huge chat burst around hour 25
    clips = [(30 * 3600 + 120, 5000), (30 * 3600 + 900, 800)]  # popular viewer clips around hour 30
    score = longform.activity_per_minute(minutes, chat, clips)
    windows = longform.pick_hot_windows(score, 120, 2)
    starts = sorted(a for a, _ in windows)
    assert len(windows) == 2 and any(1380 <= a <= 1560 for a in starts) and any(1680 <= a <= 1802 for a in starts)
    assert all(b - a == 120 for a, b in windows)
    assert longform.pick_hot_windows(np.zeros(100), 30, 2) == []


# --- worker planning -----------------------------------------------------------------------------

def _worker(tmp_path, monkeypatch, meta):
    from clipfarm import worker as wm
    from clipfarm.pipeline import download

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(download, "probe_info", lambda url, cookies="": meta)
    w = wm.Worker()
    shown = []
    w._show_status = lambda job_id, detail="", extra="": shown.append((job_id, detail, extra))
    return w, shown


def test_ten_hour_stream_is_split_into_parts(tmp_path, monkeypatch):
    w, shown = _worker(tmp_path, monkeypatch, {"duration": 10 * 3600})
    job_id = w.db.add_job("https://www.twitch.tv/videos/9", chat_id=-100, user_id=7, user_name="Anna")
    w.handle(w.db.claim_next_job())
    assert w.db.get_job(job_id)["status"] == dbm.DONE
    parts = [j for j in w.db.recent_jobs(10) if j["id"] != job_id]
    assert len(parts) == 5 and all(p["status"] == dbm.QUEUED and p["chat_id"] == -100 and p["user_name"] == "Anna"
                                   for p in parts)
    assert sorted(p["part_label"] for p in parts)[0] == "part 1/5"
    assert "processing it in 5 parts" in shown[-1][2] and "0:00:00–2:01:00" in shown[-1][2]


def test_short_range_is_processed_directly(tmp_path, monkeypatch):
    w, _ = _worker(tmp_path, monkeypatch, {"duration": 10 * 3600})
    job_id = w.db.add_job("u", range_start=3600, range_end=3600 + 7200)
    parts, _ = w.plan(w.db.get_job(job_id))
    assert parts is None


def test_long_range_is_split_within_the_range(tmp_path, monkeypatch):
    w, _ = _worker(tmp_path, monkeypatch, {"duration": 10 * 3600})
    job_id = w.db.add_job("u", range_start=3600, range_end=3600 + 5 * 3600)
    parts, note = w.plan(w.db.get_job(job_id))
    assert parts[0][0] == 3600 and parts[-1][1] == 6 * 3600 and "parts" in note


def test_very_long_stream_is_scouted(tmp_path, monkeypatch):
    from clipfarm.pipeline import twitch

    w, shown = _worker(tmp_path, monkeypatch, {"duration": 44 * 3600, "id": "v42", "extractor_key": "TwitchVod",
                                               "uploader_id": "ishowspeed"})
    rate = np.full(44 * 60 + 1, 1.0, dtype=np.float32)
    rate[25 * 60:25 * 60 + 60] = 30.0
    monkeypatch.setattr(twitch.ChatReplay, "sample_rates", lambda self, vid, duration, step=60: rate)
    monkeypatch.setattr(twitch, "public_vod_clips", lambda login, vid, ts=0.0: [
        twitch.ViewerClip(offset=30 * 3600, duration=30, views=9000, title="x", url="")])
    job_id = w.db.add_job("https://www.twitch.tv/videos/42", chat_id=-100)
    w.handle(w.db.claim_next_job())
    parts = sorted((j["range_start"], j["range_end"]) for j in w.db.recent_jobs(10) if j["id"] != job_id)
    assert len(parts) == 2 and all(b - a == 7200 for a, b in parts)
    assert any(a <= 25.5 * 3600 <= b for a, b in parts) and any(a <= 30 * 3600 <= b for a, b in parts)
    assert "hottest parts" in shown[-1][2]


def test_very_long_stream_without_data_asks_for_a_range(tmp_path, monkeypatch):
    w, shown = _worker(tmp_path, monkeypatch, {"duration": 30 * 3600, "extractor_key": "Youtube"})
    job_id = w.db.add_job("https://www.youtube.com/watch?v=x", chat_id=-100)
    failed = []
    w._fail = lambda jid, error: failed.append(error)
    w.handle(w.db.claim_next_job())
    assert failed and "time range" in failed[0]


# --- bot: range in the message ---------------------------------------------------------------------

def test_bot_stores_the_range(tmp_path):
    from clipfarm import bot
    from tests.test_group import GROUP, OWNER, _cfg, _ctx, _message

    ctx = _ctx(tmp_path, _cfg(telegram_chat_id=GROUP))
    update, replies = _message(OWNER, GROUP, "https://www.twitch.tv/videos/5 1:30:00-3:30:00", "group", first_name="Sergey")
    asyncio.run(bot.on_text(update, ctx))
    job = ctx.bot_data["db"].get_job(1)
    assert (job["range_start"], job["range_end"]) == (5400, 12600)
    assert "1:30:00–3:30:00" in replies[0]


# --- section offsets -------------------------------------------------------------------------------

def test_twitch_signals_are_shifted_into_the_section(monkeypatch):
    from clipfarm.pipeline import download, run, twitch

    cfg = SimpleNamespace(twitch_client_id="", twitch_client_secret="", twitch_chat=False)
    src = download.Source(url="https://www.twitch.tv/videos/42", video_path=None, title="", uploader="",
                          duration=0, extractor="twitchvod", video_id="v42", uploader_id="speed")
    monkeypatch.setattr(twitch, "public_vod_clips", lambda login, vid, ts=0.0: [
        twitch.ViewerClip(5500, 30, 10, "in range", ""), twitch.ViewerClip(100, 30, 99, "before", ""),
        twitch.ViewerClip(20000, 30, 5, "after", "")])
    tw = run.fetch_twitch_signals(cfg, src, 7200, offset=5400)
    assert [(c.title, c.offset) for c in tw.clips] == [("in range", 100)]
