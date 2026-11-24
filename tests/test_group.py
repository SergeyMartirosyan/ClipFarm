"""Several users and a shared group chat: who may do what, where reports go, title replies."""

import asyncio
from types import SimpleNamespace

import pytest

from clipfarm import bot, messages
from clipfarm import db as dbm

OWNER, FRIEND, STRANGER = 111, 222, 999
GROUP = -100555


def run(coro):
    return asyncio.run(coro)


def _cfg(**over):
    base = dict(telegram_owner_id=OWNER, telegram_allowed_users=(FRIEND,), telegram_chat_id=0,
                post_tz="Asia/Yerevan", post_platforms=[], posts_per_day=15,
                post_window="10:00-23:00")
    base.update(over)
    cfg = SimpleNamespace(**base)
    cfg.allowed_user_ids = {cfg.telegram_owner_id, *cfg.telegram_allowed_users}
    return cfg


class Sent:
    _next = 1000

    def __init__(self):
        Sent._next += 1
        self.message_id = Sent._next


class FakeBot:
    def __init__(self):
        self.sent = []
        self.edits = []

    async def send_message(self, chat, text, **kw):
        self.sent.append((chat, text))
        return Sent()

    async def edit_message_caption(self, **kw):
        self.edits.append(kw)


def _message(user_id, chat_id, text, chat_type, reply_to=None, first_name="Friend"):
    replies = []

    async def reply_text(t, **kw):
        replies.append(t)
        return Sent()

    msg = SimpleNamespace(
        text=text, chat_id=chat_id, chat=SimpleNamespace(type=chat_type, id=chat_id),
        reply_to_message=SimpleNamespace(message_id=reply_to) if reply_to else None, reply_text=reply_text,
    )
    user = SimpleNamespace(id=user_id, first_name=first_name, username=None)
    return SimpleNamespace(effective_message=msg, effective_user=user, effective_chat=msg.chat), replies


def _ctx(tmp_path, cfg):
    return SimpleNamespace(bot=FakeBot(), bot_data={"cfg": cfg, "db": dbm.Database(tmp_path / "t.db")})


# --- helpers ---------------------------------------------------------------------------------------

def test_allowed_users_from_env(tmp_path, monkeypatch):
    from clipfarm.config import load_config

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_OWNER_ID", "111")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "222, 333 444")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100555")
    cfg = load_config(env_file=str(tmp_path / "none.env"))
    assert cfg.allowed_user_ids == {111, 222, 333, 444} and cfg.telegram_chat_id == -100555
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "@sergey")
    with pytest.raises(SystemExit, match="TELEGRAM_ALLOWED_USERS"):
        load_config(env_file=str(tmp_path / "none.env"))


def test_title_reply_target():
    assert bot.title_reply_target(None, True, 5) is None
    assert bot.title_reply_target("7:50", True, 50) == 7         # a reply to the question
    assert bot.title_reply_target("7:50", True, 49) is None      # group chatter, not a reply
    assert bot.title_reply_target("7:50", True, None) is None
    assert bot.title_reply_target("7:50", False, None) == 7      # in a private chat any text counts


def test_report_chat():
    assert bot.report_chat(_cfg(), 42) == 42
    assert bot.report_chat(_cfg(telegram_chat_id=GROUP), 42) == GROUP


# --- links -----------------------------------------------------------------------------------------

def test_friend_posts_link_in_group(tmp_path):
    ctx = _ctx(tmp_path, _cfg(telegram_chat_id=GROUP))
    update, replies = _message(FRIEND, GROUP, "check this https://www.twitch.tv/videos/1", "supergroup")
    run(bot.on_text(update, ctx))
    job = ctx.bot_data["db"].get_job(1)
    assert job["chat_id"] == GROUP and job["user_id"] == FRIEND and job["user_name"] == "Friend"
    assert job["status"] == dbm.QUEUED and job["status_message_id"]
    assert "from Friend" in replies[0]


def test_group_only_mode_ignores_private_messages(tmp_path):
    ctx = _ctx(tmp_path, _cfg(telegram_chat_id=GROUP))
    update, replies = _message(OWNER, OWNER, "https://www.twitch.tv/videos/2", "private", first_name="Sergey")
    run(bot.on_text(update, ctx))
    assert replies == [] and ctx.bot.sent == [] and ctx.bot_data["db"].recent_jobs(5) == []
    other_group, replies = _message(FRIEND, -100777, "https://www.twitch.tv/videos/3", "group")
    run(bot.on_text(other_group, ctx))
    assert replies == [] and ctx.bot_data["db"].recent_jobs(5) == []


def test_without_group_private_chat_works(tmp_path):
    ctx = _ctx(tmp_path, _cfg())
    update, replies = _message(OWNER, OWNER, "https://www.twitch.tv/videos/2", "private", first_name="Sergey")
    run(bot.on_text(update, ctx))
    assert "from Sergey" in replies[0] and ctx.bot_data["db"].get_job(1)["chat_id"] == OWNER


def test_no_per_person_limit(tmp_path):
    ctx = _ctx(tmp_path, _cfg(telegram_chat_id=GROUP))
    for i in range(5):
        update, replies = _message(FRIEND, GROUP, f"https://twitch.tv/videos/{i}", "group")
        run(bot.on_text(update, ctx))
    assert len(ctx.bot_data["db"].recent_jobs(10)) == 5


def test_group_chatter_is_ignored(tmp_path):
    ctx = _ctx(tmp_path, _cfg(telegram_chat_id=GROUP))
    update, replies = _message(FRIEND, GROUP, "lol that stream was crazy", "group")
    run(bot.on_text(update, ctx))
    assert replies == [] and ctx.bot.sent == []


# --- buttons -----------------------------------------------------------------------------------------

def _clip(db, chat=GROUP):
    job = db.add_job("u", chat_id=chat)
    cid = db.add_clip(job_id=job, idx=1, total=3, start=0, end=30, title="Old title", hashtags="gta6")
    db.update_clip(cid, telegram_message_id=500)
    return cid


def _press(user_id, data, first_name="Friend", chat=GROUP):
    answers, edits, prompts = [], [], []

    async def answer(text=None, **kw):
        answers.append(text)

    async def edit_message_caption(**kw):
        edits.append(kw)

    async def reply_text(t, **kw):
        prompts.append((t, kw))
        return SimpleNamespace(message_id=600)

    user = SimpleNamespace(id=user_id, first_name=first_name, username=None,
                           mention_html=lambda: f"<a>{first_name}</a>")
    q = SimpleNamespace(from_user=user, data=data, answer=answer, edit_message_caption=edit_message_caption,
                        message=SimpleNamespace(video=True, reply_text=reply_text, chat=SimpleNamespace(id=chat)))
    return SimpleNamespace(callback_query=q), answers, edits, prompts


def test_friend_can_approve_and_caption_says_who(tmp_path):
    ctx = _ctx(tmp_path, _cfg(telegram_chat_id=GROUP))
    db = ctx.bot_data["db"]
    cid = _clip(db)
    update, answers, edits, _ = _press(FRIEND, f"ok:{cid}")
    run(bot.on_button(update, ctx))
    clip = db.get_clip(cid)
    assert clip["status"] == dbm.APPROVED and clip["decided_by"] == "Friend"
    assert "Approved by Friend" in edits[0]["caption"]
    update, _, edits, _ = _press(OWNER, f"undo:{cid}", "Sergey")
    run(bot.on_button(update, ctx))
    assert db.get_clip(cid)["decided_by"] == "" and "by" not in messages.clip_caption(db.get_clip(cid)).split("\n")[-1]


def test_stranger_cannot_press_buttons(tmp_path):
    ctx = _ctx(tmp_path, _cfg(telegram_chat_id=GROUP))
    db = ctx.bot_data["db"]
    cid = _clip(db)
    update, answers, edits, _ = _press(STRANGER, f"ok:{cid}")
    run(bot.on_button(update, ctx))
    assert answers == ["Not allowed."] and db.get_clip(cid)["status"] == dbm.PENDING_REVIEW


def test_edit_title_in_group_needs_a_reply(tmp_path):
    ctx = _ctx(tmp_path, _cfg(telegram_chat_id=GROUP))
    db = ctx.bot_data["db"]
    cid = _clip(db)
    update, _, _, prompts = _press(FRIEND, f"title:{cid}")
    run(bot.on_button(update, ctx))
    assert "reply to this message" in prompts[0][0] and prompts[0][1]["reply_markup"].selective
    # Someone (even the same person) just chatting in the group: no rename.
    chat, _ = _message(FRIEND, GROUP, "brb", "group")
    run(bot.on_text(chat, ctx))
    assert db.get_clip(cid)["title"] == "Old title"
    # Another allowed user replying to the prompt doesn't take over the friend's rename either.
    other, _ = _message(OWNER, GROUP, "Owner title", "group", reply_to=600)
    run(bot.on_text(other, ctx))
    assert db.get_clip(cid)["title"] == "Old title"
    # The friend replies to the question: renamed, and the clip caption is edited in the group.
    reply, replies = _message(FRIEND, GROUP, "Speed crashes the getaway car", "group", reply_to=600)
    run(bot.on_text(reply, ctx))
    assert db.get_clip(cid)["title"] == "Speed crashes the getaway car"
    assert ctx.bot.edits[0]["chat_id"] == GROUP and replies[0].startswith("✏️ Title updated")


def test_buttons_outside_the_group_are_refused(tmp_path):
    ctx = _ctx(tmp_path, _cfg(telegram_chat_id=GROUP))
    db = ctx.bot_data["db"]
    cid = _clip(db)
    update, answers, _, _ = _press(OWNER, f"ok:{cid}", "Sergey", chat=OWNER)
    run(bot.on_button(update, ctx))
    assert answers == ["Use the buttons in the group."] and db.get_clip(cid)["status"] == dbm.PENDING_REVIEW
