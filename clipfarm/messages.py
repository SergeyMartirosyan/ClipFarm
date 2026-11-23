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
    dbm.APPROVED: "✅ Approved: queued for posting",
    dbm.SKIPPED: "⏭ Skipped",
    dbm.POSTED: "📤 Posted",
}


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
    lines = [head]
    if job["source_title"]:
        lines.append(job["source_title"])
    lines.append(job["url"])
    if status == "failed" and job["error"]:
        lines.append(f"\nReason: {job['error'][:500]}")
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def clip_caption(clip: Mapping[str, Any]) -> str:
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
        lines += ["", note]
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
    "/stats: clip counts and this month's fallback spend"
)
