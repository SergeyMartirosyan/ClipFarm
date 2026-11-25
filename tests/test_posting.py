"""Posting: slot scheduling, the posts queue, the Instagram client (mocked API) and the poster."""

import dataclasses
import random
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
import pytest

from clipfarm import db as dbm
from clipfarm.config import load_config
from clipfarm.publish import Published, schedule
from clipfarm.publish.instagram import REFRESH_AFTER, Instagram, InstagramError, build_caption

TZ = "Asia/Yerevan"


def ts(s: str) -> float:
    return datetime.fromisoformat(s).replace(tzinfo=ZoneInfo(TZ)).timestamp()


def local(t: float) -> str:
    return datetime.fromtimestamp(t, ZoneInfo(TZ)).strftime("%Y-%m-%d %H:%M")


# --- slots ---------------------------------------------------------------------------------------

def test_slots_are_spread_over_the_window():
    slots = schedule.day_slots(datetime.fromtimestamp(ts("2026-11-24T09:00"), ZoneInfo(TZ)), 13,
                               schedule.parse_window("10:00-23:00"), ZoneInfo(TZ))
    assert [local(s)[-5:] for s in slots[:3]] == ["10:30", "11:30", "12:30"] and local(slots[-1])[-5:] == "22:30"


def test_next_slot_skips_taken_and_rolls_to_next_day():
    now = ts("2026-11-24T21:00")
    first = schedule.next_slot(now, [], 13, "10:00-23:00", TZ, jitter=0)
    assert local(first) == "2026-11-24 21:30"
    second = schedule.next_slot(now, [first], 13, "10:00-23:00", TZ, jitter=0)
    assert local(second) == "2026-11-24 22:30"
    third = schedule.next_slot(now, [first, second], 13, "10:00-23:00", TZ, jitter=0)
    assert local(third) == "2026-11-25 10:30"


def test_next_slot_respects_min_lead_and_jitter_stays_small():
    now = ts("2026-11-24T10:29")  # the 10:30 slot is too close
    slot = schedule.next_slot(now, [], 13, "10:00-23:00", TZ, jitter=240, rng=random.Random(3))
    assert abs(slot - ts("2026-11-24T11:30")) <= 240


def test_bad_window_is_rejected():
    with pytest.raises(ValueError):
        schedule.parse_window("23:00-10:00")


def test_describe():
    now = ts("2026-11-24T09:00")
    assert schedule.describe(ts("2026-11-24T18:40"), TZ, now) == "today 18:40"
    assert schedule.describe(ts("2026-11-25T10:05"), TZ, now) == "tomorrow 10:05"


# --- posts queue ----------------------------------------------------------------------------------

def _db_with_clip(tmp_path, status=dbm.APPROVED, **clip):
    d = dbm.Database(tmp_path / "t.db")
    job = d.add_job("u", chat_id=5)
    fields = dict(job_id=job, idx=1, total=1, start=0, end=30, title="Speed falls off the map", hashtags="gta6 ishowspeed")
    fields.update(clip)
    cid = d.add_clip(**fields)
    d.update_clip(cid, status=status, telegram_message_id=77)
    return d, cid


def test_posts_queue_lifecycle(tmp_path):
    d, cid = _db_with_clip(tmp_path)
    d.schedule_post(cid, "instagram", 1000)
    assert [r["id"] for r in d.due_posts(999)] == []
    due = d.due_posts(1000)
    assert len(due) == 1 and d.claim_post(due[0]["id"]) and not d.claim_post(due[0]["id"])
    d.update_post(due[0]["id"], status=dbm.POST_POSTED)
    d.schedule_post(cid, "instagram", 5000)  # re-approving a posted clip doesn't re-post
    assert d.posts_for_clip(cid)[0]["status"] == dbm.POST_POSTED


def test_undo_cancels_and_reapprove_reschedules(tmp_path):
    d, cid = _db_with_clip(tmp_path)
    d.schedule_post(cid, "instagram", 1000)
    assert d.cancel_posts(cid) == 1
    assert d.due_posts(2000) == [] and d.taken_slots("instagram", 0) == []
    d.schedule_post(cid, "instagram", 3000)
    p = d.posts_for_clip(cid)[0]
    assert p["status"] == dbm.POST_SCHEDULED and p["scheduled_at"] == 3000


def test_bot_books_a_slot_per_connected_platform(tmp_path, monkeypatch):
    from clipfarm.bot import schedule_clip

    d, cid = _db_with_clip(tmp_path)
    cfg = SimpleNamespace(post_platforms=["instagram"], posts_per_day=15, post_window="10:00-23:00", post_tz=TZ)
    summary = schedule_clip(d, cfg, cid)
    assert summary.startswith("✅ Posting: Instagram ")
    assert d.posts_for_clip(cid)[0]["status"] == dbm.POST_SCHEDULED
    assert schedule_clip(d, SimpleNamespace(**{**vars(cfg), "post_platforms": []}), cid) == ""


def test_caption():
    c = build_caption("Speed falls off", "gta6 #ishowspeed", "Follow for more")
    assert c == "Speed falls off\n\n#gta6 #ishowspeed\n\nFollow for more"
    assert build_caption("x" * 3000, "") == "x" * 2200


# --- Instagram client (mocked Graph API) -------------------------------------------------------

def _cfg(tmp_path, **over):
    import os

    old = os.environ.get("DATA_DIR")
    os.environ["DATA_DIR"] = str(tmp_path / "data")
    try:
        cfg = load_config(env_file=str(tmp_path / "none.env"))
    finally:
        if old is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = old
    return dataclasses.replace(cfg, **({"instagram_access_token": "IGAAT-env-token"} | over))


class Graph:
    """A tiny fake of graph.instagram.com."""

    def __init__(self, statuses=("IN_PROGRESS", "FINISHED"), fail=None):
        self.statuses = list(statuses)
        self.calls = []
        self.fail = fail or {}

    def __call__(self, req: httpx.Request):
        import re

        path = re.sub(r"^/v\d+\.\d+", "", req.url.path)
        self.hosts = getattr(self, "hosts", set()) | {req.url.host}
        params = dict(req.url.params) if req.method == "GET" else dict(httpx.QueryParams(req.content.decode()))
        self.calls.append((req.method, path, params))
        if path in self.fail:
            return httpx.Response(400, json={"error": self.fail[path]})
        if path == "/me":
            return httpx.Response(200, json={"user_id": "1789", "username": "sergey"})
        if path == "/1789" and params.get("fields") == "id,username":
            return httpx.Response(200, json={"id": "1789", "username": "los.clipos"})
        if req.url.host == "rupload.facebook.com":
            self.uploaded = (dict(req.headers), req.content)
            return httpx.Response(200, json={"success": True})
        if path == "/1789/media":
            assert params["media_type"] == "REELS"
            if params.get("upload_type") == "resumable":
                return httpx.Response(200, json={"id": "C1", "uri": "https://rupload.facebook.com/ig-api-upload/v99.0/C1"})
            assert params["video_url"].startswith("https://")
            return httpx.Response(200, json={"id": "C1"})
        if path == "/C1":
            return httpx.Response(200, json={"status_code": self.statuses.pop(0), "status": "x"})
        if path == "/1789/media_publish":
            assert params["creation_id"] == "C1"
            return httpx.Response(200, json={"id": "M9"})
        if path == "/M9":
            return httpx.Response(200, json={"permalink": "https://www.instagram.com/reel/abc/"})
        if path == "/refresh_access_token":
            assert params["grant_type"] == "ig_refresh_token"
            return httpx.Response(200, json={"access_token": "IGAAT-refreshed", "expires_in": 5184000})
        return httpx.Response(404, json={"error": {"message": "unknown path", "code": 100}})


def _ig(tmp_path, graph, clock=None):
    cfg = _cfg(tmp_path)
    d = dbm.Database(cfg.db_path)
    t = {"now": 1_000_000.0}
    ig = Instagram(cfg, d, http=httpx.Client(transport=httpx.MockTransport(graph)), sleep=lambda s: t.update(now=t["now"] + s),
                   clock=clock or (lambda: t["now"]))
    return ig, d, t


def test_publish_reel_flow(tmp_path):
    g = Graph()
    ig, _, _ = _ig(tmp_path, g)
    media_id, link = ig.publish_reel("https://pub-x.r2.dev/clips/a.mp4", "caption")
    assert (media_id, link) == ("M9", "https://www.instagram.com/reel/abc/")
    paths = [c[1] for c in g.calls]
    assert paths == ["/me", "/1789/media", "/C1", "/C1", "/1789/media_publish", "/M9"]
    assert all(c[2]["access_token"] == "IGAAT-env-token" for c in g.calls)


def test_processing_error_is_permanent(tmp_path):
    ig, _, _ = _ig(tmp_path, Graph(statuses=("IN_PROGRESS", "ERROR")))
    with pytest.raises(InstagramError) as e:
        ig.publish_reel("https://pub-x.r2.dev/a.mp4", "c")
    assert not e.value.transient


def test_error_classification(tmp_path):
    ig, _, _ = _ig(tmp_path, Graph(fail={"/me": {"message": "Invalid OAuth access token", "code": 190}}))
    with pytest.raises(InstagramError) as e:
        ig.me()
    assert not e.value.transient and "Generate a new one" in str(e.value)
    ig, _, _ = _ig(tmp_path / "b", Graph(fail={"/me": {"message": "rate limit", "code": 4}}))
    with pytest.raises(InstagramError) as e:
        ig.me()
    assert e.value.transient


def test_token_refresh_and_env_takeover(tmp_path):
    ig, d, t = _ig(tmp_path, Graph())
    assert ig.token == "IGAAT-env-token"
    assert ig.maybe_refresh() is False                      # fresh token: nothing to do
    t["now"] += REFRESH_AFTER + 1
    assert ig.maybe_refresh() is True and ig.token == "IGAAT-refreshed"
    assert ig.maybe_refresh() is False
    # The user generates a new token in the dashboard and puts it in .env: it takes over.
    ig.cfg = dataclasses.replace(ig.cfg, instagram_access_token="IGAAT-new-from-dashboard")
    assert ig.token == "IGAAT-new-from-dashboard"


# --- poster -----------------------------------------------------------------------------------------

class FakePublisher:
    def __init__(self, error=None, note=""):
        self.error = error
        self.note = note
        self.calls = []
        self.items = []

    def publish(self, item):
        url = item.public_url() if item.public_url else None
        self.calls.append((url, item.text))
        self.items.append(item)
        self.cover = item.cover_offset
        if self.error:
            raise self.error
        return Published("M1", "https://www.instagram.com/reel/xyz/", note=self.note)

    def maybe_refresh(self):
        if self.error and not self.error.transient:
            raise self.error
        return False


class FakeTg:
    def __init__(self):
        self.sent = []

    def send_message(self, chat, text, keyboard=None, reply_to=None):
        self.sent.append((chat, text, reply_to))
        return 1


class FakeR2:
    def __init__(self):
        self.uploads = []

    def upload(self, local, key, content_type):
        self.uploads.append(key)
        return f"https://pub-x.r2.dev/{key}"


def _poster(tmp_path, publisher, clip_fields=None, r2=True):
    from clipfarm.poster import Poster

    cfg = _cfg(tmp_path, r2_public_base_url="https://pub-x.r2.dev" if r2 else "",
               r2_account_id="a" if r2 else "", r2_access_key_id="k", r2_secret_access_key="s", r2_bucket="b")
    d = dbm.Database(cfg.db_path)
    video = tmp_path / "clip-01.mp4"
    video.write_bytes(b"mp4")
    job = d.add_job("u", chat_id=5)
    fields = dict(job_id=job, idx=1, total=1, start=0, end=30, title="Speed falls off", hashtags="gta6",
                  local_path=str(video))
    fields.update(clip_fields or {})
    cid = d.add_clip(**fields)
    d.update_clip(cid, status=dbm.APPROVED, telegram_message_id=77)
    d.schedule_post(cid, "instagram", 100)
    t = {"now": 200.0}
    tg, r2c = FakeTg(), FakeR2()
    p = Poster(cfg=cfg, db=d, tg=tg, publishers={"instagram": publisher}, clock=lambda: t["now"], r2=r2c if r2 else None)
    p.last_maintenance = t["now"]
    return p, d, cid, tg, r2c, t


def test_poster_posts_due_clip_and_reports(tmp_path):
    pub = FakePublisher()
    p, d, cid, tg, r2, _ = _poster(tmp_path, pub)
    assert p.tick() == 1
    assert pub.calls[0][0].startswith("https://pub-x.r2.dev/clips/") and pub.calls[0][1] == "Speed falls off\n\n#gta6"
    assert r2.uploads  # the clip had no public URL yet: uploaded on demand
    assert d.posts_for_clip(cid)[0]["status"] == dbm.POST_POSTED
    assert d.get_clip(cid)["status"] == dbm.POSTED
    assert tg.sent == [(5, "📤 Posted to Instagram: https://www.instagram.com/reel/xyz/\nSpeed falls off", 77)]
    assert p.tick() == 0


def test_poster_retries_transient_then_gives_up(tmp_path):
    pub = FakePublisher(InstagramError("Instagram: busy (code 2)", transient=True))
    p, d, cid, tg, _, t = _poster(tmp_path, pub, {"public_url": "https://pub-x.r2.dev/a.mp4"})
    p.tick()
    post = d.posts_for_clip(cid)[0]
    assert post["status"] == dbm.POST_SCHEDULED and post["attempts"] == 1 and post["next_try_at"] == 200 + 300
    assert p.tick() == 0  # not yet
    for _ in range(3):
        t["now"] = d.posts_for_clip(cid)[0]["next_try_at"] or t["now"]
        p.tick()
    post = d.posts_for_clip(cid)[0]
    assert post["status"] == dbm.POST_FAILED and post["attempts"] == 4
    assert len(pub.calls) == 4 and "Couldn't post to Instagram" in tg.sent[-1][1]


def test_poster_permanent_error_reports_once(tmp_path):
    pub = FakePublisher(InstagramError("access token invalid", transient=False))
    p, d, cid, tg, _, _ = _poster(tmp_path, pub, {"public_url": "https://pub-x.r2.dev/a.mp4"})
    p.tick()
    assert d.posts_for_clip(cid)[0]["status"] == dbm.POST_FAILED and len(tg.sent) == 1


def test_poster_without_r2_sends_the_file(tmp_path):
    pub = FakePublisher()
    p, d, cid, tg, _, _ = _poster(tmp_path, pub, r2=False)
    p.tick()
    assert pub.calls[0][0] is None and pub.items[0].local_path.name == "clip-01.mp4"
    assert d.posts_for_clip(cid)[0]["status"] == dbm.POST_POSTED


def test_poster_reports_a_missing_clip_file(tmp_path):
    pub = FakePublisher()
    p, d, cid, tg, _, _ = _poster(tmp_path, pub, r2=False)
    (tmp_path / "clip-01.mp4").unlink()
    p.tick()
    assert pub.calls == [] and "clip file is gone" in d.posts_for_clip(cid)[0]["error"]


def test_poster_skips_clip_that_was_undone(tmp_path):
    pub = FakePublisher()
    p, d, cid, tg, _, _ = _poster(tmp_path, pub)
    d.update_clip(cid, status=dbm.PENDING_REVIEW)  # undone after scheduling, before cancel ran
    p.tick()
    assert pub.calls == [] and d.posts_for_clip(cid)[0]["status"] == dbm.POST_CANCELLED


def test_poster_uses_post_caption_and_cover(tmp_path):
    pub = FakePublisher()
    p, d, cid, tg, _, _ = _poster(tmp_path, pub, {"caption": "Speed tries stealth in GTA 6. It goes badly.",
                                                   "cover_offset": 7.5, "public_url": "https://pub-x.r2.dev/a.mp4"})
    p.tick()
    assert pub.calls[0][1] == "Speed tries stealth in GTA 6. It goes badly.\n\n#gta6" and pub.cover == 7.5


def test_reel_cover_offset_is_sent_in_ms(tmp_path):
    g = Graph()
    ig, _, _ = _ig(tmp_path, g)
    ig.publish_reel("https://pub-x.r2.dev/a.mp4", "c", cover_offset=7.5)
    container = [c for c in g.calls if c[1] == "/1789/media"][0][2]
    assert container["thumb_offset"] == "7500"


def test_reel_from_file_uses_resumable_upload(tmp_path):
    g = Graph()
    ig, _, _ = _ig(tmp_path, g)
    video = tmp_path / "c.mp4"
    video.write_bytes(b"0123456789")
    assert ig.publish_reel_file(video, "caption", cover_offset=2) == ("M9", "https://www.instagram.com/reel/abc/")
    headers, body = g.uploaded
    assert body == b"0123456789" and headers["file_size"] == "10" and headers["offset"] == "0"
    assert headers["authorization"] == "OAuth IGAAT-env-token"
    container = [c for c in g.calls if c[1] == "/1789/media"][0][2]
    assert container["upload_type"] == "resumable" and "video_url" not in container


def test_poster_notes_and_quota_wait(tmp_path):
    from clipfarm.publish import PublishError

    pub = FakePublisher(note="⚠️ YouTube put it up as Private")
    p, d, cid, tg, _, _ = _poster(tmp_path, pub)
    p.tick()
    assert "Private" in tg.sent[-1][1] and "Private" in d.posts_for_clip(cid)[0]["error"]
    pub = FakePublisher(PublishError("limit", transient=True, retry_after=6 * 3600))
    p, d, cid, tg, _, t = _poster(tmp_path / "q", pub)
    p.tick()
    assert d.posts_for_clip(cid)[0]["next_try_at"] == t["now"] + 6 * 3600


def test_expired_login_warns_the_group_once(tmp_path):
    from clipfarm.publish import PublishError

    pub = FakePublisher(PublishError("YouTube login expired", transient=False, code="invalid_grant"))
    p, d, cid, tg, _, t = _poster(tmp_path, pub)
    p.cfg = dataclasses.replace(p.cfg, telegram_chat_id=-100)
    p.maintenance()
    p.maintenance()
    assert [m[:2] for m in tg.sent] == [(-100, "⚠️ Instagram: YouTube login expired")]
    t["now"] += 13 * 3600
    p.maintenance()
    assert len(tg.sent) == 2


def test_post_platforms_include_youtube_after_login(tmp_path):
    cfg = _cfg(tmp_path, instagram_access_token="")
    assert cfg.post_platforms == []
    cfg.youtube_token_path.write_text("{}")
    assert cfg.post_platforms == ["youtube"]


def test_reset_approved_cancels_posts_and_restores_review(tmp_path):
    d, cid = _db_with_clip(tmp_path)
    d.update_clip(cid, decided_by="Anna")
    d.schedule_post(cid, "instagram", 1000)
    _, other = _db_with_clip(tmp_path / "x", status=dbm.SKIPPED)
    rows = d.reset_approved()
    assert [r["id"] for r in rows] == [cid] and rows[0]["chat_id"] == 5
    clip = d.get_clip(cid)
    assert clip["status"] == dbm.PENDING_REVIEW and clip["decided_by"] == ""
    assert d.posts_for_clip(cid)[0]["status"] == dbm.POST_CANCELLED
    # Approving again books it fresh.
    d.schedule_post(cid, "instagram", 2000)
    assert d.posts_for_clip(cid)[0]["status"] == dbm.POST_SCHEDULED
    assert d.reset_approved() == []


def test_reset_posts_command_updates_telegram(tmp_path, monkeypatch):
    from clipfarm import cli, notify

    cfg = _cfg(tmp_path, telegram_token="1:x", telegram_owner_id=9)
    d = dbm.Database(cfg.db_path)
    job = d.add_job("u", chat_id=-100)
    cid = d.add_clip(job_id=job, idx=1, total=1, start=0, end=30, title="T", hashtags="gta6")
    d.update_clip(cid, status=dbm.APPROVED, telegram_message_id=55)
    d.schedule_post(cid, "instagram", 1000)
    edits = []
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(notify.Telegram, "edit_caption", lambda self, chat, mid, caption, kb=None: edits.append((chat, mid, kb)))
    monkeypatch.setattr("time.sleep", lambda s: None)
    cli.cmd_reset_posts(SimpleNamespace(yes=True))
    assert edits[0][:2] == (-100, 55) and edits[0][2]["inline_keyboard"][0][0]["callback_data"] == f"ok:{cid}"
    assert d.get_clip(cid)["status"] == dbm.PENDING_REVIEW


def test_facebook_style_token_uses_graph_facebook(tmp_path):
    g = Graph()
    cfg = _cfg(tmp_path, instagram_access_token="EAAUv4-token", instagram_user_id="1789")
    ig = Instagram(cfg, dbm.Database(cfg.db_path), http=httpx.Client(transport=httpx.MockTransport(g)),
                   sleep=lambda s: None)
    assert ig.me() == {"user_id": "1789", "username": "los.clipos"}
    video = tmp_path / "c.mp4"
    video.write_bytes(b"xx")
    assert ig.publish_reel_file(video, "caption")[0] == "M9"
    assert g.hosts == {"graph.facebook.com", "rupload.facebook.com"}
    assert ig.maybe_refresh() is False  # checked, not refreshed
    assert not any(c[1] == "/refresh_access_token" for c in g.calls)


def test_facebook_style_token_without_user_id_explains(tmp_path):
    g = Graph()
    cfg = _cfg(tmp_path, instagram_access_token="EAAUv4-token")
    ig = Instagram(cfg, dbm.Database(cfg.db_path), http=httpx.Client(transport=httpx.MockTransport(g)))
    with pytest.raises(InstagramError, match="INSTAGRAM_USER_ID") as e:
        ig.me()
    assert not e.value.transient


def test_upload_error_keeps_metas_whole_answer(tmp_path):
    g = Graph()
    real = g.__call__

    def handler(req):
        if req.url.host == "rupload.facebook.com":
            return httpx.Response(400, json={"debug_info": {"type": "ProcessingFailedError",
                                                            "message": "Request processing failed"}})
        return real(req)

    cfg = _cfg(tmp_path)
    ig = Instagram(cfg, dbm.Database(cfg.db_path), http=httpx.Client(transport=httpx.MockTransport(handler)))
    video = tmp_path / "c.mp4"
    video.write_bytes(b"xx")
    with pytest.raises(InstagramError) as e:
        ig.publish_reel_file(video, "c")
    assert "ProcessingFailedError" in str(e.value) and "HTTP 400" in str(e.value) and e.value.transient
    assert "rupload.facebook.com/ig-api-upload/v99.0/<container>" in str(e.value) and "IGAAT" not in str(e.value)


def test_retry_failed_posts_now(tmp_path, monkeypatch):
    from clipfarm import cli
    from clipfarm import poster as poster_mod

    pub = FakePublisher(InstagramError("refused", transient=False))
    p, d, cid, tg, _, t = _poster(tmp_path, pub)
    p.tick()
    assert [r["clip_id"] for r in d.failed_posts("instagram")] == [cid] and d.failed_posts("youtube") == []
    pub.error = None
    monkeypatch.setattr(poster_mod, "Poster", lambda: p)
    cli.cmd_retry_failed(SimpleNamespace(platform="instagram", first=True))
    post = d.posts_for_clip(cid)[0]
    assert post["status"] == dbm.POST_POSTED and post["attempts"] == 0 and d.failed_posts() == []


def test_schedule_line_shows_posting_and_retries():
    from clipfarm import messages

    base = {"platform": "instagram", "title": "T", "scheduled_at": ts("2026-11-25T20:51"), "attempts": 0,
            "next_try_at": None, "error": None}
    now = ts("2026-11-25T20:58")
    assert "posting now" in messages.schedule_line({**base, "status": "posting"}, TZ, now)
    retry = messages.schedule_line({**base, "status": "scheduled", "attempts": 1, "next_try_at": ts("2026-11-25T21:01"),
                                    "error": "Instagram: busy"}, TZ, now)
    assert "retry 2/4 at today 21:01: Instagram: busy" in retry
    assert "due" in messages.schedule_line({**base, "status": "scheduled"}, TZ, now)
    assert messages.schedule_line({**base, "status": "scheduled", "scheduled_at": ts("2026-11-25T21:40")}, TZ, now) \
        == "today 21:40 · Instagram · T"
