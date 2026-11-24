"""Telegram circuit breaker, bot startup backoff, job retries, delivery queue, log redaction, check."""

import json
import logging
from types import SimpleNamespace

import httpx
import pytest

from clipfarm import db as dbm
from clipfarm import notify
from clipfarm.logsetup import RedactFilter, redact

TOKEN = "123456789:AAH" + "x" * 32


# --- notify circuit breaker -----------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def _tg(handler, clock):
    tg = notify.Telegram(TOKEN, clock=clock, sleep=clock.sleep)
    tg.http = httpx.Client(transport=httpx.MockTransport(handler))
    return tg


def test_short_429_is_slept_off_inline():
    clock, calls = Clock(), []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={"ok": False, "error_code": 429, "parameters": {"retry_after": 2}})
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 7}})

    tg = _tg(handler, clock)
    assert tg.send_message(1, "hi") == 7
    assert clock.slept == [3] and tg.available


def test_login_refusal_opens_circuit_and_fails_fast():
    clock, calls = Clock(), []

    def handler(req):
        calls.append(1)
        return httpx.Response(429, json={"ok": False, "error_code": 429, "description": "Too Many Requests: retry after 31",
                                         "parameters": {"retry_after": 31}})

    tg = _tg(handler, clock)
    with pytest.raises(notify.TelegramUnavailable):
        tg.send_message(1, "hi")
    assert not tg.available and len(calls) == 1
    # While open: no request at all.
    with pytest.raises(notify.TelegramUnavailable):
        tg.edit_message(1, 2, "x")
    assert len(calls) == 1
    # Waits at least the 60 s floor (not just retry_after), with jitter.
    assert 1060 <= tg.blocked_until <= 1066 + 1


def test_timeouts_and_non_json_open_circuit_and_floor_doubles():
    clock = Clock()

    def timeout(req):
        raise httpx.ReadTimeout("slow", request=req)

    tg = _tg(timeout, clock)
    with pytest.raises(notify.TelegramUnavailable):
        tg.send_message(1, "hi")
    first = tg.blocked_until - clock.t
    clock.t = tg.blocked_until + 1
    tg.http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(502, text="<html>bad gateway</html>")))
    with pytest.raises(notify.TelegramUnavailable):
        tg.send_message(1, "hi")
    assert tg.blocked_until - clock.t > first * 1.8  # floor doubled
    # Success resets the floor.
    clock.t = tg.blocked_until + 1
    tg.http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})))
    tg.send_message(1, "hi")
    assert tg.floor == notify.FLOOR_START


def test_other_errors_do_not_open_circuit():
    clock = Clock()
    tg = _tg(lambda r: httpx.Response(400, json={"ok": False, "error_code": 400, "description": "Bad Request: chat not found"}), clock)
    with pytest.raises(notify.TelegramError) as e:
        tg.send_message(1, "hi")
    assert not isinstance(e.value, notify.TelegramUnavailable) and tg.available


def test_video_is_reopened_on_retry(tmp_path):
    clock, sizes = Clock(), []
    video = tmp_path / "c.mp4"
    video.write_bytes(b"x" * 1000)

    def handler(req):
        sizes.append(len(req.content))
        if len(sizes) == 1:
            return httpx.Response(429, json={"ok": False, "error_code": 429, "parameters": {"retry_after": 1}})
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 9}})

    tg = _tg(handler, clock)
    assert tg.send_video(1, video, "cap") == 9
    assert sizes[0] == sizes[1] > 1000  # full file both times


# --- bot startup backoff --------------------------------------------------------------------------

def test_wait_for_telegram_backs_off_then_returns(monkeypatch):
    from clipfarm import bot

    answers = [
        httpx.ReadTimeout("login in progress"),
        {"ok": False, "error_code": 429, "parameters": {"retry_after": 31}},
        {"ok": False, "error_code": 429, "parameters": {"retry_after": 200}},
        {"ok": True, "result": {"username": "GagoClipFarmBot"}},
    ]

    def fake_get(url, timeout):
        assert timeout >= 90
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return SimpleNamespace(json=lambda: a)

    monkeypatch.setattr(bot.httpx, "get", fake_get)
    slept = []
    cfg = SimpleNamespace(telegram_api_base="https://api.telegram.org", telegram_token=TOKEN)
    assert bot.wait_for_telegram(cfg, sleep=slept.append) == "GagoClipFarmBot"
    assert 60 <= slept[0] <= 66          # timeout counts like a 429: base delay
    assert 120 <= slept[1] <= 132        # delay doubled, beats retry_after 31
    assert 240 <= slept[2] <= 264        # max(200, 240)


# --- jobs: attempts, superseded clips, 'new' status, delivery queue ------------------------------

def test_requeue_supersedes_unreviewed_clips_and_caps_attempts(tmp_path):
    d = dbm.Database(tmp_path / "t.db")
    job = d.add_job("u", chat_id=5)
    assert d.claim_next_job()["attempts"] == 1
    pending = d.add_clip(job_id=job, idx=1, total=2, start=0, end=30, title="a")
    approved = d.add_clip(job_id=job, idx=2, total=2, start=40, end=70, title="b")
    d.set_clip_decision(approved, dbm.APPROVED)

    assert d.requeue_stuck_jobs() == 1
    assert d.get_clip(pending)["status"] == dbm.SUPERSEDED
    assert d.get_clip(approved)["status"] == dbm.APPROVED
    assert d.get_job(job)["status"] == dbm.QUEUED

    assert d.claim_next_job()["attempts"] == 2
    d.requeue_stuck_jobs()  # died again
    j = d.get_job(job)
    assert j["status"] == dbm.FAILED and "stopped twice" in j["error"]


def test_new_jobs_are_not_claimable_until_announced(tmp_path):
    d = dbm.Database(tmp_path / "t.db")
    j = d.add_job("u", chat_id=5, status=dbm.NEW)
    assert d.queue_position(j) == 1
    assert d.claim_next_job() is None
    d.update_job(j, status=dbm.QUEUED, status_message_id=42)
    assert d.claim_next_job()["status_message_id"] == 42


def test_undelivered_clips_queue(tmp_path):
    d = dbm.Database(tmp_path / "t.db")
    tg_job, cli_job = d.add_job("u", chat_id=5), d.add_job("v")
    a = d.add_clip(job_id=tg_job, idx=1, total=2, start=0, end=30, title="a")
    b = d.add_clip(job_id=tg_job, idx=2, total=2, start=0, end=30, title="b")
    d.add_clip(job_id=cli_job, idx=1, total=1, start=0, end=30, title="cli")
    assert [r["id"] for r in d.undelivered_clips()] == [a, b]
    d.update_clip(a, telegram_message_id=99)
    d.update_clip(b, deliver_attempts=3)
    assert d.undelivered_clips() == []


class FakeTg:
    def __init__(self, fail_from=None):
        self.sent = []
        self.edits = []
        self.available = True
        self.fail_from = fail_from

    def _maybe_fail(self):
        if self.fail_from is not None and len(self.sent) + len(self.edits) >= self.fail_from:
            self.available = False
            raise notify.TelegramUnavailable("429", 0)

    def send_video(self, chat, path, caption, keyboard):
        self._maybe_fail()
        self.sent.append(caption.splitlines()[2])
        return 100 + len(self.sent)

    def send_message(self, chat, text, keyboard=None):
        self._maybe_fail()
        self.sent.append(text)
        return 500

    def edit_message(self, chat, mid, text):
        self._maybe_fail()
        self.edits.append(text)


def _worker(tmp_path, monkeypatch, tg):
    from clipfarm import worker as wm

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    w = wm.Worker()
    w.tg = tg
    return w


def test_worker_delivers_later_when_telegram_comes_back(tmp_path, monkeypatch):
    tg = FakeTg(fail_from=1)
    w = _worker(tmp_path, monkeypatch, tg)
    job = w.db.add_job("u", chat_id=5)
    w.db.update_job(job, status_message_id=7)
    ids = [w.db.add_clip(job_id=job, idx=i, total=3, start=0, end=30, title=f"clip {i}", local_path=str(tmp_path / f"c{i}.mp4"))
           for i in (1, 2, 3)]
    w.flush_telegram()
    assert tg.sent == ["clip 1"]  # then Telegram went away: stop, don't hammer
    w._show_status(job, "rendering 3/3")
    assert job in w.stale_status and tg.edits == []

    tg.available, tg.fail_from = True, None  # circuit closed again
    w.flush_telegram()
    assert tg.sent == ["clip 1", "clip 2", "clip 3"]
    assert len(tg.edits) == 1 and not w.stale_status
    assert all(w.db.get_clip(i)["telegram_message_id"] for i in ids)


# --- logging ----------------------------------------------------------------------------------------

def test_tokens_and_secrets_are_redacted():
    line = f"HTTP Request: POST https://api.telegram.org/bot{TOKEN}/sendMessage client_secret=abc123&x=1"
    out = redact(line)
    assert TOKEN.split(":")[1] not in out and "abc123" not in out
    assert "bot123456789:***" in out

    record = logging.LogRecord("httpx", logging.INFO, "", 0, "url %s", (f"/bot{TOKEN}/getMe",), None)
    RedactFilter().filter(record)
    assert TOKEN.split(":")[1] not in record.getMessage()


# --- self-check telegram messages ---------------------------------------------------------------

@pytest.mark.parametrize("answer, expect", [
    ({"ok": True, "result": {"username": "B"}}, "OK"),
    ({"ok": False, "error_code": 429, "parameters": {"retry_after": 30}}, "refuses to log this bot in"),
    ({"ok": False, "error_code": 401, "description": "Unauthorized"}, "token rejected"),
    ({"ok": False, "error_code": 404, "description": "Not Found"}, "malformed token"),
    ("timeout", "no answer in"),
    ("html", "isn't from Telegram"),
])
def test_check_telegram_messages(monkeypatch, answer, expect):
    from clipfarm import check

    def fake_get(url, timeout):
        assert timeout >= 90
        if answer == "timeout":
            raise httpx.ReadTimeout("slow")
        if answer == "html":
            return httpx.Response(502, text="<html>Bad gateway</html>")
        return httpx.Response(200, json=answer)

    monkeypatch.setattr(check.httpx, "get", fake_get)
    check.results.clear()
    cfg = SimpleNamespace(telegram_token=TOKEN, telegram_owner_id=5, telegram_api_base="https://api.telegram.org",
                          telegram_chat_id=0, allowed_user_ids={5})
    check.check_telegram(cfg)
    level, _, detail = check.results[-1]
    assert (level == "OK") if expect == "OK" else (expect in detail and level == "FAIL")
    assert TOKEN.split(":")[1] not in detail and "bot id 123456789" in detail


def test_check_rejects_malformed_token():
    from clipfarm import check

    check.results.clear()
    check.check_telegram(SimpleNamespace(telegram_token="123:short", telegram_owner_id=5,
                                         telegram_api_base="https://api.telegram.org"))
    assert "doesn't look like a bot token" in check.results[-1][2]


# --- small fixes ------------------------------------------------------------------------------------

def test_preset_boxes_are_clamped_inside_frame():
    from clipfarm.pipeline.layout import preset_layout

    lay = preset_layout({"cam": [0.9, 0.9, 0.5, 0.5], "game": [-0.1, 0, 1.3, 1.2]}, 1920, 1080)
    for b in (lay.cam, lay.main):
        assert b.x >= 0 and b.y >= 0 and b.x + b.w <= 1920 and b.y + b.h <= 1080 and b.w % 2 == 0 and b.h % 2 == 0


def test_missing_ranking_gives_no_bonus():
    from clipfarm.pipeline import vision
    from clipfarm.pipeline.judge import Candidate

    cands = [Candidate(0, 30, "a", [], 6, 6, 6, ""), Candidate(40, 70, "b", [], 6, 6, 6, "")]
    v = vision.parse_verdicts(json.dumps({"clips": [
        {"id": 0, "what_happens": "x", "visual": 5, "hook": 5, "menu_or_idle": False},
        {"id": 1, "what_happens": "y", "visual": 5, "hook": 5, "menu_or_idle": False}], "ranking": []}), 2)
    vision.apply_verdicts(cands, v)
    assert all(c.rank_bonus == 0 for c in cands)


def test_non_numeric_owner_id_is_a_clear_error(tmp_path, monkeypatch):
    from clipfarm.config import load_config

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_OWNER_ID", "@sergey")
    with pytest.raises(SystemExit, match="TELEGRAM_OWNER_ID"):
        load_config(env_file=str(tmp_path / "none.env"))


# --- whisper gets samples, not a file path -----------------------------------------------------------

def test_transcriber_passes_samples_not_path(tmp_path, monkeypatch):
    """faster-whisper decodes file paths with PyAV, which breaks on some version pairs (av 19).
    The transcriber must hand it a float32 array of our 16 kHz WAV instead."""
    import subprocess

    import numpy as np

    from clipfarm.media import extract_audio
    from clipfarm.pipeline.transcribe import Transcriber

    src = tmp_path / "s.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=duration=2", str(src)], check=True)
    wav = extract_audio(src, tmp_path / "a.wav")
    seen = {}

    class FakeModel:
        def transcribe(self, audio, **kw):
            seen["audio"] = audio
            word = SimpleNamespace(start=0.1, end=0.4, word=" hi")
            return [SimpleNamespace(start=0.0, end=1.0, text=" hi", words=[word])], None

    t = Transcriber()
    t._model = FakeModel()
    segs = t.transcribe(wav, words=True)
    assert isinstance(seen["audio"], np.ndarray) and seen["audio"].dtype == np.float32
    assert abs(len(seen["audio"]) - 32000) < 800  # 2 s at 16 kHz
    assert segs[0].words[0].text == "hi"


def test_failed_job_keeps_download_for_retry(tmp_path, monkeypatch):
    import dataclasses

    from clipfarm.config import load_config
    from clipfarm.pipeline import run

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    cfg = load_config(env_file=str(tmp_path / "none.env"))

    def boom(url, job_id, cfg, workdir, *a):
        (workdir / "source.mp4").write_bytes(b"8 GB of stream")
        raise RuntimeError("whisper broke")

    monkeypatch.setattr(run, "_process", boom)
    with pytest.raises(RuntimeError):
        run.process("https://twitch.tv/videos/1", 1, cfg, transcriber=None)
    kept = run.work_dir_for(cfg, "https://twitch.tv/videos/1") / "source.mp4"
    assert kept.exists()  # a retry of the same link (any job id) finds it

    monkeypatch.setattr(run, "_process", lambda *a: ["ok"])
    no_keep = dataclasses.replace(cfg, keep_last_source=False)
    assert run.process("https://twitch.tv/videos/1", 2, no_keep, transcriber=None) == ["ok"]
    assert not kept.parent.exists()  # cleaned after success when KEEP_LAST_SOURCE=0


def test_last_source_is_kept_and_only_one(tmp_path, monkeypatch):
    from clipfarm.config import load_config
    from clipfarm.pipeline import run

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    cfg = load_config(env_file=str(tmp_path / "none.env"))
    assert cfg.keep_last_source

    def fake(url, job_id, cfg, workdir, *a):
        (workdir / "source.mp4").write_bytes(b"stream")
        (workdir / "clip-01.wav").write_bytes(b"tmp")
        return ["ok"]

    monkeypatch.setattr(run, "_process", fake)
    run.process("https://twitch.tv/videos/1", 1, cfg, transcriber=None)
    first = run.work_dir_for(cfg, "https://twitch.tv/videos/1")
    assert (first / "source.mp4").exists() and not (first / "clip-01.wav").exists()
    run.process("https://twitch.tv/videos/2", 2, cfg, transcriber=None)
    assert not first.exists()  # a new link replaces the kept stream
    assert (run.work_dir_for(cfg, "https://twitch.tv/videos/2") / "source.mp4").exists()


def test_transcript_cache_roundtrip(tmp_path):
    from clipfarm.pipeline import transcribe

    segs = [transcribe.Segment(1.0, 2.5, "what? what?"), transcribe.Segment(3.0, 4.0, "no way")]
    path = tmp_path / "transcript.json"
    transcribe.save_segments(segs, path)
    back = transcribe.load_segments(path)
    assert [(s.start, s.end, s.text) for s in back] == [(1.0, 2.5, "what? what?"), (3.0, 4.0, "no way")]
    assert transcribe.load_segments(tmp_path / "missing.json") == []
    (tmp_path / "bad.json").write_text("{half")
    assert transcribe.load_segments(tmp_path / "bad.json") == []


def test_check_flags_positive_group_id(monkeypatch):
    from clipfarm import check

    monkeypatch.setattr(check.httpx, "get", lambda url, timeout: httpx.Response(200, json={"ok": True, "result": {"username": "B"}}))
    check.results.clear()
    check.check_telegram(SimpleNamespace(telegram_token=TOKEN, telegram_owner_id=5, telegram_api_base="https://api.telegram.org",
                                         telegram_chat_id=5495066032, allowed_user_ids={5}))
    assert check.results[-1][0] == "FAIL" and "minus sign" in check.results[-1][2]
