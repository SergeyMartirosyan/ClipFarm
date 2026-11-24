"""Telegram texts and buttons, shared by the bot and the worker."""

from __future__ import annotations

from typing import Any, Mapping

from . import db as dbm

JOB_STATUS = {
    "queued": "⏳ Queued",
    "downloading": "⬇️ Downloading",
    "transcribing": "📝 Transcribing",
    "analyzing": "🧠 Finding the viral moments",
    "rendering": "🎞 Rendering clips",
    "done": "✅ Done",
    "failed": "❌ Failed",
}

DECISION_NOTE = {
    dbm.APPROVED: "✅ Approved",
    dbm.SKIPPED: "⏭ Skipped",
    dbm.POSTED: "📤 Posted",
    dbm.SUPERSEDED: "♻️ Replaced by a re-run",
}


def _get(row: Mapping[str, Any], key: str, default: Any = "") -> Any:
    try:
        value = row[key]
    except (KeyError, IndexError):
        return default
    return default if value is None else value


def hms(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def job_text(job: Mapping[str, Any], detail: str = "", extra: str = "") -> str:
    status = job["status"]
    head = f"{JOB_STATUS.get(status, status)} · job #{job['id']}"
    if detail:
        head += f" ({detail})"
    who = _get(job, "user_name")
    if who:
        head += f" · from {who}"
    lines = [head]
    if job["source_title"]:
        lines.append(job["source_title"])
    lines.append(job["url"])
    if status == "failed" and job["error"]:
        lines.append(f"\nReason: {job['error'][:500]}")
    if extra:
        lines.append(extra)
    return "\n".join(lines)


POST_LABEL = {"instagram": "Instagram"}


def posts_lines(posts, tz_name: str) -> list[str]:
    """One line per platform: when it goes out, or what happened."""
    from .publish.schedule import describe

    lines = []
    for p in posts:
        name = POST_LABEL.get(p["platform"], p["platform"])
        status = p["status"]
        if status in (dbm.POST_SCHEDULED, dbm.POST_POSTING):
            lines.append(f"📅 {name}: {describe(p['scheduled_at'], tz_name)}")
        elif status == dbm.POST_POSTED:
            lines.append(f"📤 {name}: posted")
        elif status == dbm.POST_FAILED:
            lines.append(f"⚠️ {name}: failed")
    return lines


def clip_caption(clip: Mapping[str, Any], posts=None, tz_name: str = "UTC") -> str:
    dur = clip["end"] - clip["start"]
    tags = " ".join(f"#{t}" for t in clip["hashtags"].split() if t)
    lines = [
        f"🎬 Clip {clip['idx']}/{clip['total']} · ⭐ {clip['final_score']:.1f}/10 · {dur:.0f}s · {clip['layout']}",
        "",
        clip["title"],
    ]
    if tags:
        lines.append(tags)
    lines += [
        "",
        f"⏱ {hms(clip['start'])}–{hms(clip['end'])} in source · judge: {clip['judge_model']}",
    ]
    if clip["reason"]:
        lines.append(f"💡 {clip['reason']}")
    if clip["vision_note"]:
        lines.append(f"👁 {clip['vision_note']}")
    note = DECISION_NOTE.get(clip["status"])
    if note:
        who = _get(clip, "decided_by")
        if who and clip["status"] in (dbm.APPROVED, dbm.SKIPPED):
            note += f" by {who}"
        lines += ["", note]
    if posts:
        lines += posts_lines(posts, tz_name)
    return "\n".join(lines)[:1024]  # Telegram caption limit


def clip_buttons(clip: Mapping[str, Any]) -> list[list[tuple[str, str]]]:
    """Rows of (label, callback_data)."""
    cid = clip["id"]
    if clip["status"] == dbm.PENDING_REVIEW:
        return [[("✅ Post", f"ok:{cid}"), ("✏️ Edit title", f"title:{cid}"), ("⏭ Skip", f"skip:{cid}")]]
    if clip["status"] in (dbm.APPROVED, dbm.SKIPPED):
        return [[("↩️ Undo", f"undo:{cid}")]]
    return []


def buttons_as_json(rows: list[list[tuple[str, str]]]) -> dict:
    return {"inline_keyboard": [[{"text": t, "callback_data": d} for t, d in row] for row in rows]}


HELP = (
    "Send me a video link (YouTube, Twitch VOD, Kick, …) and I'll cut it into vertical clips.\n\n"
    "Each finished clip arrives here with buttons:\n"
    "✅ Post · ✏️ Edit title · ⏭ Skip\n\n"
    "Commands:\n"
    "/queue: recent jobs and their status\n"
    "/stats: clip counts and this month's fallback spend\n"
    "/schedule: upcoming posts\n\n"
    "In a group, just paste the link. To change a title, tap ✏️ and reply to the bot's question.\n"
    "/chatid: show this chat's id (for TELEGRAM_CHAT_ID)"
)
