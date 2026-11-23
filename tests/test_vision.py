"""Vision review: frame sampling (real ffmpeg), prompt building, scoring, fail-soft paths."""

import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from clipfarm.pipeline import vision
from clipfarm.pipeline.judge import Candidate
from clipfarm.pipeline.transcribe import Segment


def _cand(start, end, hook=6, title="t"):
    return Candidate(start, end, title, [], hook, 6, 6, "")


def test_parse_verdicts_ranks_and_validates():
    raw = json.dumps({
        "clips": [
            {"id": 0, "what_happens": "menu screen", "visual": 1, "hook": 2, "menu_or_idle": True},
            {"id": 1, "what_happens": "clutch 1v3", "visual": 9, "hook": 15, "menu_or_idle": False},
            {"id": 7, "what_happens": "bogus id", "visual": 5, "hook": 5, "menu_or_idle": False},
        ],
        "ranking": [1, 1, 0, 9],
    })
    v = vision.parse_verdicts(raw, count=2)
    assert set(v) == {0, 1}
    assert v[1].rank == 0 and v[0].rank == 1
    assert v[1].hook == 10  # clamped
    assert v[0].menu_or_idle is True


def test_apply_verdicts_moves_scores_the_right_way():
    clutch, menu, talk = _cand(0, 30), _cand(40, 70), _cand(80, 110)
    for c in (clutch, menu, talk):
        c.signal_score = 0.6
    before = {id(c): c.final_score for c in (clutch, menu, talk)}
    verdicts = {
        0: vision.Verdict("clutch play", visual=9, hook=9, menu_or_idle=False, rank=0),
        1: vision.Verdict("settings menu", visual=1, hook=2, menu_or_idle=True, rank=2),
        2: vision.Verdict("just chatting", visual=4, hook=5, menu_or_idle=False, rank=1),
    }
    vision.apply_verdicts([clutch, menu, talk], verdicts)
    assert clutch.final_score > before[id(clutch)]
    assert menu.final_score < before[id(menu)] - 3
    assert clutch.final_score > talk.final_score > menu.final_score
    assert clutch.rank_bonus == pytest.approx(1.0) and menu.rank_bonus == pytest.approx(-1.0)
    assert clutch.vision_note == "clutch play"


def test_unreviewed_candidates_keep_their_scores():
    c = _cand(0, 30)
    s = c.final_score
    vision.apply_verdicts([c], {})
    assert c.final_score == s and c.visual is None


def test_build_parts_interleaves_text_and_images():
    from google.genai import types

    cands = [_cand(0, 30, title="first"), _cand(40, 70, title="second")]
    cands[0].extra["audience"] = "chat 80 msgs/12s, mostly funny (KEKW ×70)"
    segs = [Segment(1, 5, "no way he did that"), Segment(45, 50, "what")]
    parts = vision.build_parts(cands, [[b"jpg1", b"jpg2"], [b"jpg3"]], segs, types)
    texts = [p.text for p in parts if p.text]
    images = [p for p in parts if p.inline_data]
    assert len(images) == 3
    assert "Clip id 0" in texts[1] and "no way he did that" in texts[1] and "KEKW" in texts[1]
    assert "Clip id 1" in texts[2] and "Audience: no data" in texts[2]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_sample_frames_returns_jpegs(tmp_path):
    src = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=20",
                    "-c:v", "libx264", "-preset", "ultrafast", str(src)], check=True)
    frames = vision.sample_frames(src, 2, 18, n=4)
    assert len(frames) == 4 and all(f[:2] == b"\xff\xd8" for f in frames)


class FakeJudge:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

        def generate_content(model, contents, config):
            self.calls.append(model)
            r = self.responses[model]
            if isinstance(r, Exception):
                raise r
            return SimpleNamespace(text=r)

        self.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    def _gemini_client(self):
        return self.client

    def _gemini_chain(self):
        return list(self.responses)


def _reviewer(monkeypatch, responses):
    monkeypatch.setattr(vision, "sample_frames", lambda video, s, e, n=6: [b"\xff\xd8x"])
    judge = FakeJudge(responses)
    return vision.VisionReviewer(SimpleNamespace(gemini_api_key="k"), judge), judge


def test_review_falls_through_to_next_model_and_applies(monkeypatch):
    good = json.dumps({"clips": [{"id": 0, "what_happens": "a", "visual": 2, "hook": 2, "menu_or_idle": True},
                                 {"id": 1, "what_happens": "b", "visual": 9, "hook": 9, "menu_or_idle": False}],
                       "ranking": [1, 0]})
    reviewer, judge = _reviewer(monkeypatch, {"busy-model": RuntimeError("503 UNAVAILABLE"), "ok-model": good})
    cands = [_cand(0, 30), _cand(40, 70)]
    note = reviewer.review("v.mp4", cands, [])
    assert judge.calls == ["busy-model", "ok-model"]
    assert "AI watched 2 candidates (ok-model), 1 menu/idle" == note
    assert cands[1].final_score > cands[0].final_score


def test_review_failure_keeps_scores(monkeypatch):
    reviewer, _ = _reviewer(monkeypatch, {"m": RuntimeError("boom")})
    cands = [_cand(0, 30), _cand(40, 70)]
    before = [c.final_score for c in cands]
    assert "failed" in reviewer.review("v.mp4", cands, [])
    assert [c.final_score for c in cands] == before


def test_review_skipped_without_key():
    r = vision.VisionReviewer(SimpleNamespace(gemini_api_key=""), None)
    assert "skipped" in r.review("v.mp4", [_cand(0, 30)], [])
