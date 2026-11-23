"""Telegram bot process: receives links, handles Post / Edit title / Skip.

    python -m clipfarm.bot
"""

from __future__ import annotations

import logging
import re

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from . import db as dbm
from . import messages
from .config import load_config

log = logging.getLogger("clipfarm.bot")

URL_RE = re.compile(r"https?://[^\s<>\"']+")
AWAITING_TITLE = "awaiting_title_clip_id"


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
        lines.append(f"#{j['id']} {label}: {name[:70]}")
    await update.effective_message.reply_text("\n".join(lines), disable_web_page_preview=True)


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
    msg = update.effective_message
    text = msg.text or ""
    urls = URL_RE.findall(text)

    pending = db.kv_get(AWAITING_TITLE)
    if pending and not urls:
        clip_id = int(pending)
        db.kv_set(AWAITING_TITLE, None)
        new_title = " ".join(text.split())[:100]
        if not new_title:
            await msg.reply_text("Empty title, kept the old one.")
            return
        db.update_clip(clip_id, title=new_title)
        clip = db.get_clip(clip_id)
        if clip and clip["telegram_message_id"]:
            try:
                await ctx.bot.edit_message_caption(
                    chat_id=msg.chat_id,
                    message_id=clip["telegram_message_id"],
                    caption=messages.clip_caption(clip),
                    reply_markup=_markup(messages.clip_buttons(clip)),
                )
            except BadRequest as exc:  # e.g. the clip was sent as a text message
                log.info("Caption edit failed: %s", exc)
        await msg.reply_text(f"✏️ Title updated: {new_title}")
        return

    if not urls:
        await msg.reply_text("Send me a video link (YouTube, Twitch VOD, Kick, …). /start for help.")
        return

    for url in urls:
        job_id = db.add_job(url, msg.chat_id)
        pos = db.queue_position(job_id)
        job = db.get_job(job_id)
        detail = "next up" if pos <= 1 else f"position {pos}"
        reply = await msg.reply_text(messages.job_text(job, detail), disable_web_page_preview=True)
        db.update_job(job_id, status_message_id=reply.message_id)


async def on_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    cfg = ctx.bot_data["cfg"]
    if q.from_user is None or q.from_user.id != cfg.telegram_owner_id:
        await q.answer("Not allowed.", show_alert=True)
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

    if action == "title":
        db.kv_set(AWAITING_TITLE, str(clip["id"]))
        await q.answer()
        await q.message.reply_text(f"Send the new title for clip {clip['idx']}/{clip['total']}:")
        return

    new_status = {"ok": dbm.APPROVED, "skip": dbm.SKIPPED, "undo": dbm.PENDING_REVIEW}.get(action)
    if new_status is None:
        await q.answer()
        return
    if new_status == dbm.PENDING_REVIEW:
        db.update_clip(clip["id"], status=new_status, decided_at=None)
    else:
        db.set_clip_decision(clip["id"], new_status)
    clip = db.get_clip(clip["id"])
    await q.answer(messages.DECISION_NOTE.get(new_status, "Back to review"))
    try:
        if q.message and q.message.video:
            await q.edit_message_caption(
                caption=messages.clip_caption(clip), reply_markup=_markup(messages.clip_buttons(clip))
            )
        else:
            await q.edit_message_reply_markup(reply_markup=_markup(messages.clip_buttons(clip)))
    except BadRequest as exc:
        log.info("Message edit failed: %s", exc)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # it logs every poll, including the token URL
    cfg = load_config()
    if not cfg.telegram_enabled:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_OWNER_ID in .env")

    app = Application.builder().token(cfg.telegram_token).build()
    app.bot_data["cfg"] = cfg
    app.bot_data["db"] = dbm.Database(cfg.db_path)

    owner = filters.User(user_id=cfg.telegram_owner_id)
    app.add_handler(CommandHandler(["start", "help"], cmd_start, filters=owner))
    app.add_handler(CommandHandler("queue", cmd_queue, filters=owner))
    app.add_handler(CommandHandler("stats", cmd_stats, filters=owner))
    app.add_handler(MessageHandler(owner & filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(CallbackQueryHandler(on_button))

    log.info("Bot started; only user %s can use it", cfg.telegram_owner_id)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
