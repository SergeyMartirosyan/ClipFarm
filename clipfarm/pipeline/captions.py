"""Burned-in captions: short uppercase word groups, the spoken word highlighted."""

from __future__ import annotations

from pathlib import Path

from .transcribe import Word

# ASS colours are &HAABBGGRR
WHITE = "&H00FFFFFF"
BLACK = "&H00000000"
HIGHLIGHT = "&H0000E5FF"  # warm yellow

MAX_WORDS = 3
MAX_CHARS = 22
GAP_BREAK = 0.6  # a pause this long starts a new group


def group_words(words: list[Word]) -> list[list[Word]]:
    groups: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if cur:
            text_len = sum(len(x.text) + 1 for x in cur) + len(w.text)
            pause = w.start - cur[-1].end
            ends_sentence = cur[-1].text[-1:] in ".?!,"
            if len(cur) >= MAX_WORDS or text_len > MAX_CHARS or pause > GAP_BREAK or ends_sentence:
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def _ts(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").upper()


def build_ass(words: list[Word], margin_v: int = 640, font: str = "DejaVu Sans") -> str:
    """Word times must be relative to the clip start."""
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},78,{WHITE},{WHITE},{BLACK},&H80000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines: list[str] = []
    groups = group_words(words)
    for gi, group in enumerate(groups):
        group_end = group[-1].end
        next_start = groups[gi + 1][0].start if gi + 1 < len(groups) else None
        if next_start is not None and next_start - group_end < 0.3:
            group_end = next_start  # avoid flicker between close groups
        for wi, word in enumerate(group):
            start = word.start
            end = group[wi + 1].start if wi + 1 < len(group) else group_end
            if end <= start:
                continue
            parts = [
                f"{{\\c{HIGHLIGHT}&}}{_escape(w.text)}{{\\c{WHITE}&}}" if j == wi else _escape(w.text)
                for j, w in enumerate(group)
            ]
            lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Cap,,0,0,0,,{' '.join(parts)}")
    return header + "\n".join(lines) + "\n"


def write_ass(words: list[Word], path: Path, margin_v: int = 640) -> Path:
    Path(path).write_text(build_ass(words, margin_v=margin_v), encoding="utf-8")
    return Path(path)
