"""Telegram bot process: receives links, handles Post / Edit title / Skip.

    python -m clipfarm.bot
"""

from __future__ import annotations

import logging
import random
import re
import time

import httpx

from telegram import ForceReply, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from . import db as dbm
from . import messages
from .config import load_config
from .logsetup import setup_logging

log = logging.getLogger("clipfarm.bot")

URL_RE = re.compile(r"https?://[^\s<>\"']+")
AWAITING_TITLE = "awaiting_title:"  # + user id -> "clip_id:prompt_message_id"


def in_home_chat(cfg, chat_id: int | None) -> bool:
    """With TELEGRAM_CHAT_ID set, the bot only works in that group; otherwise anywhere (setup)."""
    return not cfg.telegram_chat_id or chat_id == cfg.telegram_chat_id


def report_chat(cfg, chat_id: int) -> int:
    """Where a job's status, clips and posting reports go: the shared group if one is set."""
    return int(cfg.telegram_chat_id or chat_id)


def display_name(user) -> str:
    if user is None:
        return ""
    return (user.first_name or user.username or str(user.id))[:40]


def title_reply_target(pending: str | None, is_group: bool, reply_to_id: int | None) -> int | None:
    """The clip whose title this message sets, or None. In a group only an actual reply to the
    bot's question counts, so normal chatting never renames a clip by accident."""
    if not pending:
        return None
    clip_id, _, prompt_id = pending.partition(":")
    if is_group and (not prompt_id or reply_to_id != int(prompt_id)):
        return None
    return int(clip_id)


def _markup(rows: list[list[tuple[str, str]]]) -> InlineKeyboardMarkup | None:
    if not rows:
        return None
    return InlineKeyboardMarkup([[InlineKeyboardButton(t, callback_data=d) for t, d in row] for row in rows])


def _db(ctx: ContextTypes.DEFAULT_TYPE) -> dbm.Database:
    return ctx.bot_data["db"]


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(messages.HELP)


async def cmd_queue(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    jobs = _db(ctx).recent_jobs(10)
    if not jobs:
        await update.effective_message.reply_text("No jobs yet. Send me a link.")
        return
    lines = []
    for j in jobs:
        label = messages.JOB_STATUS.get(j["status"], j["status"])
        name = j["source_title"] or j["url"]
        who = f" ({j['user_name']})" if j["user_name"] else ""
        lines.append(f"#{j['id']} {label}{who}: {name[:70]}")
    await update.effective_message.reply_text("\n".join(lines), disable_web_page_preview=True)


def schedule_clip(db: dbm.Database, cfg, clip_id: int) -> str:
    """Book the next free slot on every connected platform. Returns a short summary."""
    from .publish.schedule import describe, next_slot

    now = time.time()
    booked = []
    for platform in cfg.post_platforms:
        at = next_slot(now, db.taken_slots(platform, now - 86400), cfg.posts_per_day, cfg.post_window, cfg.post_tz)
        db.schedule_post(clip_id, platform, at)
        booked.append(f"{messages.POST_LABEL.get(platform, platform)} {describe(at, cfg.post_tz)}")
    return ("✅ Posting: " + ", ".join(booked)) if booked else ""


async def cmd_schedule(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    from .publish.schedule import describe

    cfg = ctx.bot_data["cfg"]
    rows = _db(ctx).upcoming_posts(20)
    if not cfg.post_platforms:
        await update.effective_message.reply_text("No platform connected yet (set INSTAGRAM_ACCESS_TOKEN in .env).")
        return
    if not rows:
        await update.effective_message.reply_text("Nothing scheduled. Approve clips with ✅ Post.")
        return
    lines = [f"{describe(r['scheduled_at'], cfg.post_tz)} · {messages.POST_LABEL.get(r['platform'], r['platform'])} · "
             f"{r['title'][:60]}" for r in rows]
    await update.effective_message.reply_text("Upcoming posts:\n" + "\n".join(lines))


async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    db = _db(ctx)
    counts = db.clip_counts()
    cfg = ctx.bot_data["cfg"]
    spent = db.spend_this_month("anthropic")
    text = (
        f"Clips awaiting review: {counts.get(dbm.PENDING_REVIEW, 0)}\n"
        f"Approved (waiting to post): {counts.get(dbm.APPROVED, 0)}\n"
        f"Posted: {counts.get(dbm.POSTED, 0)}\n"
        f"Skipped: {counts.get(dbm.SKIPPED, 0)}\n\n"
        f"Fallback LLM spend this month: ${spent:.3f} of ${cfg.fallback_monthly_cap_usd:.2f} cap"
    )
    await update.effective_message.reply_text(text)


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    db = _db(ctx)
    cfg = ctx.bot_data["cfg"]
    msg = update.effective_message
    user = update.effective_user
    text = msg.text or ""
    urls = URL_RE.findall(text)
    is_group = msg.chat.type in ("group", "supergroup")
    if not in_home_chat(cfg, msg.chat_id):
        return  # group-only mode: private messages (and other groups) are ignored

    key = AWAITING_TITLE + str(user.id)
    clip_id = title_reply_target(db.kv_get(key), is_group,
                                 msg.reply_to_message.message_id if msg.reply_to_message else None)
    if clip_id is not None and not urls:
        db.kv_set(key, None)
        new_title = " ".join(text.split())[:100]
        if not new_title:
            await msg.reply_text("Empty title, kept the old one.")
            return
        db.update_clip(clip_id, title=new_title)
        clip = db.get_clip(clip_id)
        job = db.get_job(clip["job_id"]) if clip else None
        if clip and clip["telegram_message_id"]:
            try:
                await ctx.bot.edit_message_caption(
                    chat_id=(job["chat_id"] if job and job["chat_id"] else msg.chat_id),
                    message_id=clip["telegram_message_id"],
                    caption=messages.clip_caption(clip, db.posts_for_clip(clip_id), cfg.post_tz),
                    reply_markup=_markup(messages.clip_buttons(clip)),
                )
            except BadRequest as exc:  # e.g. the clip was sent as a text message
                log.info("Caption edit failed: %s", exc)
        await msg.reply_text(f"✏️ Title updated: {new_title}")
        return

    if not urls:
        if not is_group:  # in a group, people just chat: stay quiet
            await msg.reply_text("Send me a video link (YouTube, Twitch VOD, Kick, …). /start for help.")
        return

    for url in urls:
        chat = report_chat(cfg, msg.chat_id)
        # 'new' until the status message exists, so the worker can't claim the job and post a
        # second status message while Telegram is slow to answer this reply.
        job_id = db.add_job(url, chat, status=dbm.NEW, user_id=user.id, user_name=display_name(user))
        pos = db.queue_position(job_id)
        shown = dict(db.get_job(job_id)) | {"status": dbm.QUEUED}
        detail = "next up" if pos <= 1 else f"position {pos}"
        message_id = None
        try:
            if chat == msg.chat_id:
                sent = await msg.reply_text(messages.job_text(shown, detail), disable_web_page_preview=True)
            else:
                sent = await ctx.bot.send_message(chat, messages.job_text(shown, detail), disable_web_page_preview=True)
                await msg.reply_text(f"Queued as job #{job_id}. Updates and clips go to the group.")
            message_id = sent.message_id
        finally:
            db.update_job(job_id, status=dbm.QUEUED, status_message_id=message_id)


async def cmd_chatid(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    note = "" if chat.type == "private" else "\nPut this in .env as TELEGRAM_CHAT_ID, then restart the bot and worker."
    await update.effective_message.reply_text(f"Chat id: {chat.id} ({chat.type}){note}")


async def on_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    cfg = ctx.bot_data["cfg"]
    if q.from_user is None or q.from_user.id not in cfg.allowed_user_ids:
        await q.answer("Not allowed.", show_alert=True)
        return
    if not in_home_chat(cfg, q.message.chat.id if q.message and q.message.chat else None):
        await q.answer("Use the buttons in the group.", show_alert=True)
        return
    db = _db(ctx)
    try:
        action, raw_id = (q.data or "").split(":", 1)
        clip = db.get_clip(int(raw_id))
    except ValueError:
        clip = None
    if clip is None:
        await q.answer("Clip not found.")
        return

    if clip["status"] == dbm.POSTED:
        await q.answer("Already posted.")
        return
    if clip["status"] == dbm.SUPERSEDED:
        await q.answer("This clip is from an interrupted run; the job was re-run, see the newer clips.", show_alert=True)
        return

    if action == "title":
        await q.answer()
        prompt = await q.message.reply_text(
            f"{q.from_user.mention_html()}, reply to this message with the new title for clip "
            f"{clip['idx']}/{clip['total']}:",
            parse_mode="HTML",
            reply_markup=ForceReply(selective=True),
        )
        db.kv_set(AWAITING_TITLE + str(q.from_user.id), f"{clip['id']}:{prompt.message_id}")
        return

    new_status = {"ok": dbm.APPROVED, "skip": dbm.SKIPPED, "undo": dbm.PENDING_REVIEW}.get(action)
    if new_status is None:
        await q.answer()
        return
    if new_status == dbm.PENDING_REVIEW:
        db.update_clip(clip["id"], status=new_status, decided_at=None, decided_by="")
        db.cancel_posts(clip["id"])
    else:
        db.set_clip_decision(clip["id"], new_status)
        db.update_clip(clip["id"], decided_by=display_name(q.from_user))
    answer = messages.DECISION_NOTE.get(new_status, "Back to review")
    if new_status == dbm.APPROVED:
        answer = schedule_clip(db, cfg, clip["id"]) or "✅ Approved (no platform connected yet)"
    clip = db.get_clip(clip["id"])
    await q.answer(answer)
    try:
        if q.message and q.message.video:
            await q.edit_message_caption(
                caption=messages.clip_caption(clip, db.posts_for_clip(clip["id"]), cfg.post_tz),
                reply_markup=_markup(messages.clip_buttons(clip)),
            )
        else:
            await q.edit_message_reply_markup(reply_markup=_markup(messages.clip_buttons(clip)))
    except BadRequest as exc:
        log.info("Message edit failed: %s", exc)


def wait_for_telegram(cfg, sleep=time.sleep) -> str:
    """Block until Telegram has logged the bot in, then return its username.

    A 429 on getMe means Telegram refused to log the bot in; a ~30 s hang is a login attempt in
    progress. Each request made after the countdown ends starts a new login, so probes are spaced
    out: max(retry_after, delay) with delay doubling 60 s -> 30 min, plus jitter. Exiting instead
    would let systemd restart the bot every few seconds and keep the bot locked out."""
    delay = 60.0
    while True:
        why = ""
        retry_after = 0
        try:
            resp = httpx.get(f"{cfg.telegram_api_base}/bot{cfg.telegram_token}/getMe", timeout=90)
            body = resp.json()
            if body.get("ok"):
                return body["result"]["username"]
            if body.get("error_code") == 401:
                why = "token rejected (401): fix TELEGRAM_BOT_TOKEN in .env"
                retry_after = 600
            else:
                why = str(body.get("description"))
                retry_after = int((body.get("parameters") or {}).get("retry_after") or 0)
        except httpx.HTTPError as exc:
            why = f"no answer ({exc.__class__.__name__}): Telegram may be logging the bot in"
        except ValueError:
            why = "reply was not from Telegram (check TELEGRAM_API_BASE)"
        wait = max(retry_after, delay) * random.uniform(1.0, 1.1)
        log.warning("Telegram not ready: %s. Next try in %.0fs.", why, wait)
        sleep(wait)
        delay = min(delay * 2, 1800.0)


def main() -> None:
    setup_logging()
    cfg = load_config()
    if not cfg.telegram_enabled:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_OWNER_ID in .env")

    app = (
        Application.builder()
        .token(cfg.telegram_token)
        .base_url(f"{cfg.telegram_api_base}/bot")
        .base_file_url(f"{cfg.telegram_api_base}/file/bot")
        .connect_timeout(30)
        .read_timeout(60)
        .write_timeout(60)
        .get_updates_read_timeout(60)
        .build()
    )
    app.bot_data["cfg"] = cfg
    app.bot_data["db"] = dbm.Database(cfg.db_path)

    users = filters.User(user_id=sorted(cfg.allowed_user_ids))  # owner + TELEGRAM_ALLOWED_USERS
    # Group-only mode: everything except /chatid works in the configured group only.
    owner = users & (filters.Chat(chat_id=cfg.telegram_chat_id) if cfg.telegram_chat_id else filters.ALL)
    app.add_handler(CommandHandler(["start", "help"], cmd_start, filters=owner))
    app.add_handler(CommandHandler("chatid", cmd_chatid, filters=users))
    app.add_handler(CommandHandler("queue", cmd_queue, filters=owner))
    app.add_handler(CommandHandler("stats", cmd_stats, filters=owner))
    app.add_handler(CommandHandler("schedule", cmd_schedule, filters=owner))
    app.add_handler(MessageHandler(owner & filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(CallbackQueryHandler(on_button))

    if cfg.telegram_chat_id > 0:
        log.warning("TELEGRAM_CHAT_ID=%s is positive, but group ids are negative (e.g. -%s). The bot will "
                    "ignore the group until it is fixed in .env.", cfg.telegram_chat_id, cfg.telegram_chat_id)
    username = wait_for_telegram(cfg)
    log.info("Bot @%s started; users allowed: %s; reports to %s", username,
             ", ".join(str(u) for u in sorted(cfg.allowed_user_ids)),
             f"group {cfg.telegram_chat_id} only" if cfg.telegram_chat_id else "private chats (no group set)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
