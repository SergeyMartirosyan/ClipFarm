"""Gemini looks at the shortlist, then ranks it.

Everything before this step is a stand-in for the picture: the transcript, chat, loudness. In a
gaming stream the moment is often purely visual (a clutch play, a fail, a boss going down), and
a loud stretch can just as well be a menu or a loading screen. So the best candidates are shown
to Gemini as a handful of frames each, together with what was said and how chat reacted, and it:

  * scores how much actually happens on screen (visual) and how strong the opening is (hook),
  * flags menus, loading screens, idle/AFK stretches,
  * ranks all of them against each other. Comparative ranking is more reliable than the
    absolute 0-10 scores from the first pass, which tend to cluster.

One request for the whole shortlist (frames are ~260 tokens each). Fails soft: if anything goes
wrong, the first-pass scores are kept.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .judge import Candidate
from .transcribe import Segment

log = logging.getLogger(__name__)

FRAMES_PER_CLIP = 4
FRAME_WIDTH = 384          # ≤384 px images cost Gemini the minimum (~258 tokens each)
REQUEST_TIMEOUT_S = 150    # sent to Google as the server deadline too
TOTAL_BUDGET_S = 420       # give up on the review (keep transcript scores) after this

VISION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "what_happens": {"type": "string"},
                    "visual": {"type": "integer"},
                    "hook": {"type": "integer"},
                    "menu_or_idle": {"type": "boolean"},
                },
                "required": ["id", "what_happens", "visual", "hook", "menu_or_idle"],
            },
        },
        "ranking": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["clips", "ranking"],
}

SYSTEM_PROMPT = """You review candidate short-form clips cut from a long stream or video. For each \
candidate you get evenly spaced frames from the clip, what was said, and how the audience reacted.

For each clip:
- what_happens: one short sentence describing what is visibly going on.
- visual (0-10): how much happens on screen. A fight, a goal, a boss kill, a fail, a big reaction \
on the face cam = high. Static screen, talking with nothing happening = low. Talking-head content \
is not punished for being visually calm if the face shows a strong reaction.
- hook (0-10): would the first 2 seconds (the first frame and the first words) stop someone \
scrolling?
- menu_or_idle: true only if most frames are a menu, settings, loading screen, queue, lobby, \
desktop, or the streamer is away.

Then ranking: every clip id, best first, judged as a viewer scrolling TikTok who has never seen \
this streamer. Prefer clips with a clear moment, a reaction, and a payoff."""


@dataclass
class Verdict:
    what_happens: str
    visual: int
    hook: int
    menu_or_idle: bool
    rank: int | None  # 0 = best; None when the model didn't rank this clip


def sample_frames(video: Path, start: float, end: float, n: int = FRAMES_PER_CLIP) -> list[bytes]:
    """n JPEG frames spread over [start, end] (fast seek, one tiny ffmpeg call per frame)."""
    frames: list[bytes] = []
    for i in range(n):
        t = start + (end - start) * (i + 0.5) / n
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video),
                "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH}:-2", "-q:v", "5",
                "-f", "image2pipe", "-vcodec", "mjpeg", "-",
            ],
            capture_output=True,
        )
        if proc.returncode == 0 and proc.stdout:
            frames.append(proc.stdout)
    return frames


def _said(segments: list[Segment], start: float, end: float, limit: int = 400) -> str:
    text = " ".join(s.text for s in segments if s.end > start and s.start < end)
    return (text[:limit] + "…") if len(text) > limit else text


def build_parts(candidates: list[Candidate], frames: list[list[bytes]], segments: list[Segment], types) -> list:
    """Interleaved text + image parts: one block per candidate."""
    parts: list = [types.Part.from_text(text=f"{len(candidates)} candidate clips follow.")]
    for i, (c, imgs) in enumerate(zip(candidates, frames)):
        header = (
            f"\n--- Clip id {i} ---\n"
            f"Length: {c.end - c.start:.0f}s. Proposed title: {c.title}\n"
            f"Said: {_said(segments, c.start, c.end) or '(nothing)'}\n"
            f"Audience: {c.extra.get('audience') or 'no data'}\n"
            f"Frames ({len(imgs)}, in order):"
        )
        parts.append(types.Part.from_text(text=header))
        parts += [types.Part.from_bytes(data=img, mime_type="image/jpeg") for img in imgs]
    return parts


def parse_verdicts(raw: str | dict, count: int) -> dict[int, Verdict]:
    data = json.loads(raw) if isinstance(raw, str) else raw
    ranking = [r for r in data.get("ranking", []) if isinstance(r, int) and 0 <= r < count]
    seen: list[int] = []
    for r in ranking:
        if r not in seen:
            seen.append(r)
    rank_of = {cid: pos for pos, cid in enumerate(seen)}
    out: dict[int, Verdict] = {}
    for item in data.get("clips", []):
        try:
            cid = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= cid < count:
            continue
        out[cid] = Verdict(
            what_happens=str(item.get("what_happens", "")).strip()[:200],
            visual=_clamp(item.get("visual")),
            hook=_clamp(item.get("hook")),
            menu_or_idle=bool(item.get("menu_or_idle", False)),
            rank=rank_of.get(cid),
        )
    return out


def _clamp(v) -> int:
    try:
        return max(0, min(10, int(round(float(v)))))
    except (TypeError, ValueError):
        return 5


def apply_verdicts(candidates: list[Candidate], verdicts: dict[int, Verdict]) -> None:
    """Fold the review into each candidate's score. Unreviewed candidates are left untouched."""
    ranked = max(1, sum(1 for v in verdicts.values() if v.rank is not None) - 1)
    for i, c in enumerate(candidates):
        v = verdicts.get(i)
        if v is None:
            continue
        c.visual = v.visual
        c.hook = round((c.hook + v.hook) / 2)
        c.menu_or_idle = v.menu_or_idle
        c.vision_note = v.what_happens
        # Best +1.0 … worst -1.0, linear in rank.
        c.rank_bonus = 0.0 if v.rank is None or ranked < 1 or len(verdicts) < 2 else 1.0 - 2.0 * min(v.rank, ranked) / ranked


class VisionReviewer:
    def __init__(self, cfg, judge):
        self.cfg = cfg
        self.judge = judge  # reuses its Gemini client and model chain

    def review(self, video: Path, candidates: list[Candidate], segments: list[Segment], clock=None) -> str:
        """Re-scores `candidates` in place. Returns a short status note.

        Each attempt is cheaper than the last: a timeout or overload usually means the request
        was too heavy for the model at that moment, so retries send fewer candidates and frames.
        Thinking is set low: the task is perception, and long thinking on dozens of images is
        what made the first real run time out (504 after 300 s)."""
        import time as _time

        clock = clock or _time.monotonic
        if not self.cfg.gemini_api_key or not candidates:
            return "vision review skipped (no Gemini key)"
        from google.genai import types

        frames = [sample_frames(video, c.start, c.end) for c in candidates]
        if not any(frames):
            return "vision review skipped (could not read frames)"
        client = self.judge._gemini_client()
        started = clock()
        errors: list[str] = []
        chain = self.judge._gemini_chain()
        for attempt, model in enumerate(chain):
            if clock() - started > TOTAL_BUDGET_S:
                errors.append("time budget used up")
                break
            # Attempt 0: everything. Later: the best half (at least 4), 3 frames each.
            n = len(candidates) if attempt == 0 else max(4, len(candidates) // (2 ** attempt))
            keep = 4 if attempt == 0 else 3
            subset = candidates[:n]
            parts = build_parts(subset, [f[:keep] for f in frames[:n]], segments, types)
            for thinking in (True, False):
                cfg = dict(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_json_schema=VISION_SCHEMA,
                    temperature=0.3,
                    http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000),
                )
                if thinking:
                    cfg["thinking_config"] = types.ThinkingConfig(thinking_level="LOW")
                try:
                    resp = client.models.generate_content(
                        model=model,
                        contents=[types.Content(role="user", parts=parts)],
                        config=types.GenerateContentConfig(**cfg),
                    )
                    verdicts = parse_verdicts(resp.text or "{}", len(subset))
                except Exception as exc:
                    msg = str(exc)
                    if thinking and ("thinking" in msg.lower() or "INVALID_ARGUMENT" in msg):
                        continue  # this model doesn't take the thinking setting: same request without it
                    errors.append(f"{model}: {msg[:120]}")
                    log.warning("Vision review with %s (%d clips) failed: %s", model, n, msg[:300])
                    break
                if verdicts:
                    apply_verdicts(subset, verdicts)
                    idle = sum(1 for v in verdicts.values() if v.menu_or_idle)
                    took = clock() - started
                    return (f"AI watched {len(verdicts)} of {len(candidates)} candidates ({model}, {took:.0f}s)"
                            + (f", {idle} menu/idle" if idle else ""))
                errors.append(f"{model}: empty review")
                break
        log.warning("Vision review failed: %s", "; ".join(errors))
        return "vision review failed, kept transcript scores"
