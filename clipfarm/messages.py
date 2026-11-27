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
    part = _get(job, "part_label")
    rs, re_ = _get(job, "range_start", None), _get(job, "range_end", None)
    if rs is not None and re_ is not None:
        head += f" · {part + ' · ' if part else ''}{hms(rs)}–{hms(re_)}"
    lines = [head]
    if job["source_title"]:
        lines.append(job["source_title"])
    lines.append(job["url"])
    if status == "failed" and job["error"]:
        lines.append(f"\nReason: {job['error'][:500]}")
    if extra:
        lines.append(extra)
    return "\n".join(lines)


POST_LABEL = {"instagram": "Instagram", "youtube": "YouTube", "tiktok": "TikTok"}


def schedule_line(row: Mapping[str, Any], tz_name: str, now: float | None = None) -> str:
    """One /schedule line: when, where, what, and whether it is being posted or retried."""
    import time

    from .publish.schedule import describe

    now = time.time() if now is None else now
    name = POST_LABEL.get(row["platform"], row["platform"])
    line = f"{describe(row['scheduled_at'], tz_name, now)} · {name} · {row['title'][:60]}"
    attempts = _get(row, "attempts", 0)
    if row["status"] == dbm.POST_POSTING:
        line += "\n   ⏳ posting now"
    elif attempts:
        retry_at = _get(row, "next_try_at", None) or row["scheduled_at"]
        line += f"\n   🔁 retry {attempts + 1}/4 at {describe(retry_at, tz_name, now)}"
        error = _get(row, "error")
        if error:
            line += f": {error[:120]}"
    elif row["scheduled_at"] < now - 120:
        line += "\n   ⏳ due, the poster picks it up within a minute"
    return line


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


NUMBERS = ["1️⃣", "2️⃣", "3️⃣"]
CAPTION_LIMIT = 1024  # Telegram's limit for a video caption


def text_options(clip: Mapping[str, Any]) -> list[dict]:
    """The clip's text options [{"overlay", "caption"}], or [] when it has only one."""
    import json

    raw = _get(clip, "text_options")
    try:
        opts = json.loads(raw) if raw else []
    except ValueError:
        return []
    return [o for o in opts if isinstance(o, dict)][:3] if len(opts) > 1 else []


def _options_block(clip: Mapping[str, Any], opts: list[dict], caption_chars: int) -> list[str]:
    chosen = int(_get(clip, "text_choice", 0) or 0)
    lines = ["✏️ Text options (tap 1–3 to switch):"]
    for n, o in enumerate(opts):
        mark = " ✅" if n == chosen else ""
        cap = o.get("caption", "") if caption_chars > 0 else ""
        cap = cap if len(cap) <= caption_chars else cap[: caption_chars - 1].rstrip() + "…"
        lines.append(f"{NUMBERS[n]}{mark} 🔝 {o.get('overlay', '')}")
        if cap:
            lines.append(f"      📝 {cap}")
    return lines


def clip_caption(clip: Mapping[str, Any], posts=None, tz_name: str = "UTC", note: str = "") -> str:
    """The clip's Telegram caption. With text options, all of them are listed (the chosen one
    marked); their post texts are shortened as needed to stay within Telegram's limit."""
    opts = text_options(clip)
    for chars in (400, 200, 120, 70, 0):
        text = _clip_caption(clip, posts, tz_name, opts, chars, status_note=note)
        if len(text) <= CAPTION_LIMIT:
            return text
    return text[:CAPTION_LIMIT]


def _clip_caption(clip, posts, tz_name, opts, caption_chars, status_note="") -> str:
    dur = clip["end"] - clip["start"]
    tags = " ".join(f"#{t}" for t in clip["hashtags"].split() if t)
    rank = _get(clip, "rank", None)
    lines = [
        f"🎬 Clip {clip['idx']}/{clip['total']} · 🤖 AI {clip['llm_score']:.1f}/10"
        + (f" · #{rank} of {clip['total']}" if rank else "") + f" · {dur:.0f}s · {clip['layout']}",
        "",
        clip["title"],
    ]
    if opts:
        lines += _options_block(clip, opts, caption_chars)
    else:
        overlay = _get(clip, "overlay_text")
        if overlay:
            lines.append(f"🔝 On video: {overlay}")
        post_text = _get(clip, "caption")
        if post_text:
            lines.append(f"📝 {post_text}")
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
    if status_note:
        lines += ["", status_note]
    return "\n".join(lines)


def clip_buttons(clip: Mapping[str, Any]) -> list[list[tuple[str, str]]]:
    """Rows of (label, callback_data)."""
    cid = clip["id"]
    if clip["status"] == dbm.PENDING_REVIEW:
        rows = [[("✅ Post", f"ok:{cid}"), ("✏️ Edit title", f"title:{cid}"), ("⏭ Skip", f"skip:{cid}")]]
        opts = text_options(clip)
        if opts:
            chosen = int(_get(clip, "text_choice", 0) or 0)
            rows.append([(f"{NUMBERS[n]}{' ✅' if n == chosen else ''}", f"txt:{cid}:{n}") for n in range(len(opts))])
        return rows
    if clip["status"] in (dbm.APPROVED, dbm.SKIPPED):
        return [[("↩️ Undo", f"undo:{cid}")]]
    return []


def buttons_as_json(rows: list[list[tuple[str, str]]]) -> dict:
    return {"inline_keyboard": [[{"text": t, "callback_data": d} for t, d in row] for row in rows]}


HELP = (
    "Send me a video link (YouTube, Twitch VOD, Kick, …) and I'll cut it into vertical clips.\n\n"
    "Each finished clip arrives here with buttons:\n"
    "✅ Post · ✏️ Edit title · ⏭ Skip\n"
    "1️⃣ 2️⃣ 3️⃣: three text options (on-video text + post caption). 1 is used unless you pick another;\n"
    "a different on-video text is put on the clip in about a minute.\n\n"
    "Commands:\n"
    "/queue: recent jobs and their status\n"
    "/stats: clip counts and this month's fallback spend\n"
    "/schedule: upcoming posts\n\n"
    "In a group, just paste the link. To change a title, tap ✏️ and reply to the bot's question.\n"
    "Only part of a long stream: add a time range after the link, e.g. <link> 1:30:00-3:30:00.\n"
    "Long streams are split into parts automatically; very long ones are scouted for their best hours.\n"
    "/chatid: show this chat's id (for TELEGRAM_CHAT_ID)"
)
