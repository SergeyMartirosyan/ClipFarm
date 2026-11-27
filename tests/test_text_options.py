"""Three text options per clip: shown in Telegram, switched with 1/2/3, hook swapped on the video."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipfarm import bot, messages
from clipfarm import db as dbm
from tests.test_group import GROUP, FRIEND, _cfg, _ctx, _press

OPTS = [
    {"overlay": "He would NOT stay down", "caption": "Bro just kept coming 💀 IShowSpeed vs the toughest NPC in GTA 6"},
    {"overlay": "POV: the NPC has plot armor", "caption": "Why won't he die 😭 Speed in GTA 6"},
    {"overlay": "He would NOT stay down", "caption": "GTA 6 NPCs are built different 🔥"},  # same hook, other text
]


def _clip(db, tmp_path, **over):
    final, base = tmp_path / "clip-01.mp4", tmp_path / "clip-01.base.mp4"
    final.write_bytes(b"final")
    base.write_bytes(b"base")
    job = db.add_job("u", chat_id=GROUP)
    fields = dict(job_id=job, idx=1, total=3, start=0, end=42, title="Speed vs the NPC", hashtags="gta6",
                  layout="facecam", local_path=str(final), base_path=str(base), text_options=json.dumps(OPTS),
                  overlay_text=OPTS[0]["overlay"], caption=OPTS[0]["caption"], public_url="https://old/x.mp4",
                  r2_key="k")
    fields.update(over)
    cid = db.add_clip(**fields)
    db.update_clip(cid, telegram_message_id=500)
    return cid


def _app_ctx(tmp_path):
    ctx = _ctx(tmp_path, _cfg(telegram_chat_id=GROUP))
    tasks = []
    ctx.application = SimpleNamespace(create_task=lambda coro: tasks.append(coro))
    return ctx, tasks


def test_caption_lists_options_and_stays_within_telegrams_limit(tmp_path):
    db = dbm.Database(tmp_path / "t.db")
    cid = _clip(db, tmp_path)
    text = messages.clip_caption(db.get_clip(cid))
    assert "1️⃣ ✅ 🔝 He would NOT stay down" in text and "2️⃣ 🔝 POV: the NPC has plot armor" in text
    long = [{"overlay": f"Hook {i}", "caption": "x" * 600} for i in range(3)]
    db.update_clip(cid, text_options=json.dumps(long), reason="r" * 200, vision_note="v" * 200)
    text = messages.clip_caption(db.get_clip(cid))
    assert len(text) <= 1024 and "3️⃣ 🔝 Hook 2" in text
    rows = messages.clip_buttons(db.get_clip(cid))
    assert [b[1] for b in rows[1]] == [f"txt:{cid}:0", f"txt:{cid}:1", f"txt:{cid}:2"] and rows[1][0][0] == "1️⃣ ✅"
    db.update_clip(cid, text_options="")
    assert len(messages.clip_buttons(db.get_clip(cid))) == 1  # one text only: no option row


def test_same_hook_other_caption_switches_without_rendering(tmp_path):
    ctx, tasks = _app_ctx(tmp_path)
    db = ctx.bot_data["db"]
    cid = _clip(db, tmp_path)
    update, answers, edits, _ = _press(FRIEND, f"txt:{cid}:2")
    asyncio.run(bot.on_button(update, ctx))
    clip = db.get_clip(cid)
    assert clip["text_choice"] == 2 and clip["caption"] == OPTS[2]["caption"] and not tasks
    assert answers == ["Option 3 selected"] and "3️⃣ ✅" in edits[-1]["caption"]


def test_other_hook_rerenders_and_replaces_the_video(tmp_path, monkeypatch):
    from clipfarm.pipeline import render

    ctx, tasks = _app_ctx(tmp_path)
    db = ctx.bot_data["db"]
    cid = _clip(db, tmp_path)
    calls = []

    def fake_rerender(base, hook, kind, duration, out):
        calls.append((Path(base).name, hook, kind, duration))
        Path(out).write_bytes(b"new final")

    monkeypatch.setattr(render, "rerender_hook", fake_rerender)
    monkeypatch.setattr(render, "render_preview", lambda final, out: Path(out).write_bytes(b"preview"))
    update, answers, edits, _ = _press(FRIEND, f"txt:{cid}:1")
    media = []

    async def edit_message_media(media=None, reply_markup=None):
        media_obj = media
        globals()["_media"] = media_obj
        edits.append({"media": media_obj, "reply_markup": reply_markup})

    update.callback_query.edit_message_media = edit_message_media
    asyncio.run(bot.on_button(update, ctx))
    assert "putting the new text" in answers[0] and "⏳" in edits[-1]["caption"] and len(tasks) == 1
    # A second tap while it renders is turned away.
    again, answers2, _, _ = _press(FRIEND, f"txt:{cid}:0")
    asyncio.run(bot.on_button(again, ctx))
    assert "one moment" in answers2[0]
    asyncio.run(tasks[0])
    assert calls == [("clip-01.base.mp4", "POV: the NPC has plot armor", "facecam", 42)]
    clip = db.get_clip(cid)
    assert clip["overlay_text"] == OPTS[1]["overlay"] and clip["public_url"] is None and clip["r2_key"] is None
    assert edits[-1]["media"] is not None and cid not in ctx.bot_data["rerendering"]


def test_failed_rerender_keeps_the_old_text(tmp_path, monkeypatch):
    from clipfarm.pipeline import render

    ctx, tasks = _app_ctx(tmp_path)
    db = ctx.bot_data["db"]
    cid = _clip(db, tmp_path)

    def boom(*a):
        raise RuntimeError("ffmpeg failed")

    monkeypatch.setattr(render, "rerender_hook", boom)
    update, _, edits, _ = _press(FRIEND, f"txt:{cid}:1")
    asyncio.run(bot.on_button(update, ctx))
    asyncio.run(tasks[0])
    clip = db.get_clip(cid)
    assert clip["text_choice"] == 0 and clip["overlay_text"] == OPTS[0]["overlay"]
    assert "Couldn't change the video text" in edits[-1]["caption"]


def test_text_can_only_change_before_approval(tmp_path):
    ctx, tasks = _app_ctx(tmp_path)
    db = ctx.bot_data["db"]
    cid = _clip(db, tmp_path, status=dbm.APPROVED)
    update, answers, _, _ = _press(FRIEND, f"txt:{cid}:1")
    asyncio.run(bot.on_button(update, ctx))
    assert "Undo" in answers[0] and db.get_clip(cid)["text_choice"] == 0 and not tasks


def test_posting_uses_the_chosen_text(tmp_path):
    from tests.test_posting import FakePublisher, _poster

    pub = FakePublisher()
    p, d, cid, tg, _, _ = _poster(tmp_path, pub, {"caption": OPTS[1]["caption"], "overlay_text": OPTS[1]["overlay"],
                                                  "public_url": "https://pub-x.r2.dev/a.mp4"})
    p.tick()
    assert pub.calls[0][1].startswith("Why won't he die 😭")
