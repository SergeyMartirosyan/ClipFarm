"""The 'brain': an LLM reads the transcript + hot zones and proposes clips.

Gemini (free tier) first; Claude Haiku as a paid fallback with a monthly cap.
Both get the same prompt and JSON schema, so the rest of the pipeline doesn't care.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field

from ..db import APPROVED, SKIPPED, Database
from .signals import HotZone

log = logging.getLogger(__name__)

# Claude Haiku 5.5 list price, USD per million tokens (prompts up to 100K tokens).
HAIKU_PRICE_IN = 0.10
HAIKU_PRICE_OUT = 0.50

CLIP_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number", "description": "Start time in seconds"},
                    "end": {"type": "number", "description": "End time in seconds"},
                    "title": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                    "hook": {"type": "integer", "description": "0-10"},
                    "standalone": {"type": "integer", "description": "0-10"},
                    "payoff": {"type": "integer", "description": "0-10"},
                    "reason": {"type": "string"},
                },
                "required": ["start", "end", "title", "hashtags", "hook", "standalone", "payoff", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a senior short-form video editor. You pick the moments from a long video \
that will perform best as standalone vertical clips on YouTube Shorts, TikTok and Instagram Reels.

A great clip:
- HOOK: the first 3 seconds grab attention (a bold claim, a question, a reaction, a surprise).
- STANDALONE: makes sense to someone who has not seen the rest of the video.
- PAYOFF: ends on a punchline, reveal, strong opinion, emotional peak or satisfying conclusion.
- Starts at the beginning of a sentence and ends after a sentence finishes.

Avoid: intros/outros, sponsor reads, "like and subscribe", dead air, moments that need earlier context, \
and anything that only works with visuals you cannot infer from the transcript.

Hot zones are seconds where the audience reacted (viewer clips, chat bursts, replay spikes, loud \
audio). Treat them as strong evidence of a moment, but not as cut points: a good clip starts with the \
build-up a few seconds before the reaction and ends once the reaction lands.

You only have the transcript, so you cannot see the screen. For streams and gaming, the best moments are \
often visual (a clutch play, a fail, a jump scare, a glitch). Viewer clip titles and chat reactions \
(KEKW/LUL = funny, Pog/W = hype, monkaS/WTF = shock) tell you what happened on screen: use them. \
Moments real viewers clipped, especially popular ones, are proven; strongly prefer them, and borrow \
the wording of a good viewer clip title when it fits.

Scores are integers 0-10. Titles are short (max 70 characters), curiosity-driven, no clickbait lies, \
no emojis. Hashtags: 3-5, lowercase, without the # sign. Reason: one sentence on why it works."""


@dataclass
class Candidate:
    start: float
    end: float
    title: str
    hashtags: list[str]
    hook: int
    standalone: int
    payoff: int
    reason: str
    model: str = ""
    signal_score: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def llm_score(self) -> float:
        return 0.4 * self.hook + 0.3 * self.standalone + 0.3 * self.payoff

    @property
    def final_score(self) -> float:
        """0..10: half the LLM's judgment, half what the audience signals say."""
        return 0.5 * self.llm_score + 0.5 * (10.0 * self.signal_score)


class JudgeError(RuntimeError):
    pass


class JudgeBudgetExceeded(JudgeError):
    """Gemini failed and the fallback's monthly cap is used up: retry the job later."""


def build_user_prompt(
    *,
    title: str,
    uploader: str,
    duration: float,
    transcript: str,
    zones: list[HotZone],
    want: int,
    min_len: int,
    max_len: int,
    approved_examples: list[str],
    skipped_examples: list[str],
    viewer_clips: list | None = None,
) -> str:
    parts = [
        f"Video title: {title or 'unknown'}",
        f"Channel: {uploader or 'unknown'}",
        f"Duration: {duration:.0f} seconds",
        "",
        f"Return up to {want} clips, best first, each {min_len}-{max_len} seconds long, not overlapping.",
        "Use the [seconds] timestamps from the transcript for start and end.",
    ]
    if approved_examples or skipped_examples:
        parts += ["", "The channel owner's taste, from past decisions:"]
        parts += [f"  APPROVED: {e}" for e in approved_examples]
        parts += [f"  SKIPPED: {e}" for e in skipped_examples]
    if viewer_clips:
        parts += ["", "Moments viewers clipped (start-end seconds, views, viewer's title), most viewed first:"]
        parts += [f"  {c.offset:.0f}-{c.end:.0f}s  {c.views} views  \"{c.title}\"" for c in viewer_clips[:40]]
    if zones:
        parts += ["", "Hot zones (second, strength 0-1, why):"]
        for z in zones:
            line = f"  {z.t:.0f}s  {z.score:.2f}  {', '.join(z.reasons) or 'combined'}"
            if z.notes:
                line += "  | " + "; ".join(z.notes)
            parts.append(line)
    parts += ["", "Transcript:", transcript]
    return "\n".join(parts)


def taste_examples(db: Database | None) -> tuple[list[str], list[str]]:
    if db is None:
        return [], []

    def fmt(row) -> str:
        return f'"{row["title"]}" ({row["reason"]})'

    return (
        [fmt(r) for r in db.recent_decisions(APPROVED, 5)],
        [fmt(r) for r in db.recent_decisions(SKIPPED, 5)],
    )


def parse_candidates(raw: str | dict, duration: float, model: str) -> list[Candidate]:
    data = json.loads(raw) if isinstance(raw, str) else raw
    out: list[Candidate] = []
    for item in data.get("clips", []):
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= start < end <= duration + 1):
            continue
        out.append(
            Candidate(
                start=start,
                end=min(end, duration),
                title=str(item.get("title", "")).strip()[:100] or "Untitled clip",
                hashtags=[str(h).strip().lstrip("#").lower() for h in item.get("hashtags", []) if str(h).strip()][:5],
                hook=_clamp_int(item.get("hook")),
                standalone=_clamp_int(item.get("standalone")),
                payoff=_clamp_int(item.get("payoff")),
                reason=str(item.get("reason", "")).strip(),
                model=model,
            )
        )
    return out


def _clamp_int(v) -> int:
    try:
        return max(0, min(10, int(round(float(v)))))
    except (TypeError, ValueError):
        return 0


def _gemini_schema(schema: dict) -> dict:
    """Gemini's schema support is narrower; drop keys it may reject."""
    s = copy.deepcopy(schema)

    def strip(node):
        if isinstance(node, dict):
            node.pop("additionalProperties", None)
            for v in node.values():
                strip(v)
        elif isinstance(node, list):
            for v in node:
                strip(v)

    strip(s)
    return s


class Judge:
    def __init__(self, cfg, db: Database | None = None):
        self.cfg = cfg
        self.db = db

    # -- providers -------------------------------------------------------------

    def _call_gemini(self, prompt: str) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.cfg.gemini_api_key)
        resp = client.models.generate_content(
            model=self.cfg.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_json_schema=_gemini_schema(CLIP_SCHEMA),
                temperature=0.4,
            ),
        )
        if self.db is not None and resp.usage_metadata is not None:
            u = resp.usage_metadata
            self.db.record_llm_usage(
                "gemini", self.cfg.gemini_model,
                int(u.prompt_token_count or 0), int(u.candidates_token_count or 0), 0.0,
            )
        if not resp.text:
            raise JudgeError("Gemini returned an empty response")
        return resp.text

    def _call_claude(self, prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self.cfg.anthropic_api_key)
        resp = client.messages.create(
            model=self.cfg.fallback_model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": CLIP_SCHEMA}},
        )
        cost = (resp.usage.input_tokens * HAIKU_PRICE_IN + resp.usage.output_tokens * HAIKU_PRICE_OUT) / 1e6
        if self.db is not None:
            self.db.record_llm_usage(
                "anthropic", self.cfg.fallback_model,
                resp.usage.input_tokens, resp.usage.output_tokens, cost,
            )
        if resp.stop_reason == "refusal":
            raise JudgeError("Claude declined this transcript")
        if resp.stop_reason == "max_tokens":
            raise JudgeError("Claude response was cut off (max_tokens)")
        text = next((b.text for b in resp.content if b.type == "text"), "")
        if not text:
            raise JudgeError("Claude returned no text")
        return text

    # -- public ------------------------------------------------------------------

    def pick(
        self,
        *,
        title: str,
        uploader: str,
        duration: float,
        transcript: str,
        zones: list[HotZone],
        viewer_clips: list | None = None,
    ) -> list[Candidate]:
        approved, skipped = taste_examples(self.db)
        want = max(self.cfg.clips_per_video + 4, int(self.cfg.clips_per_video * 1.5))
        prompt = build_user_prompt(
            title=title, uploader=uploader, duration=duration, transcript=transcript, zones=zones,
            want=want, min_len=self.cfg.clip_min_seconds, max_len=self.cfg.clip_max_seconds,
            approved_examples=approved, skipped_examples=skipped, viewer_clips=viewer_clips,
        )

        errors: list[str] = []
        if self.cfg.gemini_api_key:
            for attempt in range(3):
                try:
                    cands = parse_candidates(self._call_gemini(prompt), duration, self.cfg.gemini_model)
                    if cands:
                        return cands
                    errors.append("gemini: no valid clips in response")
                except Exception as exc:  # network, quota, bad JSON: all mean "try again / fall back"
                    errors.append(f"gemini: {exc}")
                    log.warning("Gemini attempt %d failed: %s", attempt + 1, exc)
                if attempt < 2:
                    time.sleep(20 * (attempt + 1))

        if not self.cfg.anthropic_api_key:
            raise JudgeError("; ".join(errors) or "No judge API key configured")
        if self.db is not None:
            spent = self.db.spend_this_month("anthropic")
            if spent >= self.cfg.fallback_monthly_cap_usd:
                raise JudgeBudgetExceeded(
                    f"Fallback cap reached (${spent:.2f} of ${self.cfg.fallback_monthly_cap_usd:.2f} this month)"
                )
        try:
            cands = parse_candidates(self._call_claude(prompt), duration, self.cfg.fallback_model)
        except Exception as exc:
            errors.append(f"claude: {exc}")
            raise JudgeError("; ".join(errors)) from exc
        if not cands:
            raise JudgeError("; ".join(errors + ["claude: no valid clips in response"]))
        return cands
