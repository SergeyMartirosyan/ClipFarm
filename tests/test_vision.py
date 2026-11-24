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
    assert texts[2].startswith("Clip frames (2")
    assert "Clip id 1" in texts[3] and "Audience: no data" in texts[3]
    with_before = vision.build_parts(cands, [[b"a"], [b"b"]], segs, types, before=[[b"p1", b"p2"], []])
    t2 = [p.text for p in with_before if p.text]
    assert any("BEFORE the clip" in t for t in t2) and sum(1 for p in with_before if p.inline_data) == 4


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
    assert note.startswith("AI watched 2 of 2 candidates (ok-model,") and note.endswith("1 menu/idle")
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


def _cands(n):
    return [_cand(i * 100, i * 100 + 30, title=f"c{i}") for i in range(n)]


def test_retries_send_smaller_requests_and_low_thinking(monkeypatch):
    sizes, thinking = [], []

    class Judge:
        def __init__(self):
            def gen(model, contents, config):
                parts = contents[0].parts
                sizes.append(sum(1 for p in parts if p.inline_data))
                thinking.append(config.thinking_config is not None)
                assert config.http_options.timeout == vision.REQUEST_TIMEOUT_S * 1000
                if model != "m3":
                    raise RuntimeError("504 DEADLINE_EXCEEDED")
                return SimpleNamespace(text=json.dumps({"clips": [
                    {"id": i, "what_happens": "x", "visual": 5, "hook": 5, "menu_or_idle": False} for i in range(4)],
                    "ranking": [0, 1, 2, 3]}))
            self.client = SimpleNamespace(models=SimpleNamespace(generate_content=gen))

        def _gemini_client(self):
            return self.client

        def _gemini_chain(self):
            return ["m1", "m2", "m3"]

    monkeypatch.setattr(vision, "sample_frames", lambda video, s, e, n=4: [b"\xff\xd8"] * n)
    cands = _cands(16)
    note = vision.VisionReviewer(SimpleNamespace(gemini_api_key="k"), Judge()).review("v", cands, [])
    # all, then best half, then best quarter (min 4); +2 "before" frames per clip (not for the clip at 0 s)
    assert sizes == [16 * 4 + 15 * 2, 8 * 3 + 7 * 2, 4 * 3 + 3 * 2]
    assert all(thinking)
    assert "AI watched 4 of 16" in note
    assert cands[0].visual == 5 and cands[10].visual is None


def test_thinking_setting_dropped_for_models_that_reject_it(monkeypatch):
    calls = []

    class Judge:
        def __init__(self):
            def gen(model, contents, config):
                calls.append(config.thinking_config is not None)
                if config.thinking_config is not None:
                    raise RuntimeError("400 INVALID_ARGUMENT: thinking_level is not supported")
                return SimpleNamespace(text=json.dumps({"clips": [
                    {"id": 0, "what_happens": "x", "visual": 7, "hook": 5, "menu_or_idle": False}], "ranking": [0]}))
            self.client = SimpleNamespace(models=SimpleNamespace(generate_content=gen))

        def _gemini_client(self):
            return self.client

        def _gemini_chain(self):
            return ["m1"]

    monkeypatch.setattr(vision, "sample_frames", lambda video, s, e, n=4: [b"\xff\xd8"])
    cands = _cands(2)
    assert "AI watched 1 of 2" in vision.VisionReviewer(SimpleNamespace(gemini_api_key="k"), Judge()).review("v", cands, [])
    assert calls == [True, False]


def test_review_stops_at_time_budget(monkeypatch):
    t = {"now": 0.0}
    calls = []

    class Judge:
        def __init__(self):
            def gen(model, contents, config):
                calls.append(model)
                t["now"] += vision.TOTAL_BUDGET_S + 1  # one slow failure eats the whole budget
                raise RuntimeError("504 DEADLINE_EXCEEDED")
            self.client = SimpleNamespace(models=SimpleNamespace(generate_content=gen))

        def _gemini_client(self):
            return self.client

        def _gemini_chain(self):
            return ["m1", "m2", "m3"]

    monkeypatch.setattr(vision, "sample_frames", lambda video, s, e, n=4: [b"\xff\xd8"])
    note = vision.VisionReviewer(SimpleNamespace(gemini_api_key="k"), Judge()).review(
        "v", _cands(6), [], clock=lambda: t["now"] + 1)
    assert calls == ["m1"] and "failed" in note


def test_overload_does_not_shrink_the_request(monkeypatch):
    sizes = []

    class Judge:
        def __init__(self):
            def gen(model, contents, config):
                sizes.append(sum(1 for p in contents[0].parts if p.inline_data))
                if model == "m1":
                    raise RuntimeError("503 UNAVAILABLE high demand")
                return SimpleNamespace(text=json.dumps({"clips": [], "ranking": []}))
            self.client = SimpleNamespace(models=SimpleNamespace(generate_content=gen))

        def _gemini_client(self):
            return self.client

        def _gemini_chain(self):
            return ["m1", "m2"]

    monkeypatch.setattr(vision, "sample_frames", lambda video, s, e, n=4: [b"\xff\xd8"] * n)
    vision.VisionReviewer(SimpleNamespace(gemini_api_key="k"), Judge()).review("v", _cands(10), [])
    assert sizes[0] == sizes[1]  # 503 = busy, not too big: same full request to the next model


def test_starts_late_moves_start_back_within_max_length():
    c = _cand(100, 130)
    v = vision.parse_verdicts(json.dumps({"clips": [{"id": 0, "what_happens": "car already crashed", "visual": 6,
        "hook": 5, "menu_or_idle": False, "starts_late": True, "start_earlier": 8}], "ranking": [0]}), 1)
    vision.apply_verdicts([c], v, max_len=60)
    assert c.start == 92 and c.extra["started_earlier"] == 8
    c2 = _cand(100, 155)
    vision.apply_verdicts([c2], v, max_len=60)
    assert c2.start == 95  # capped: 60 s max
    c3 = _cand(100, 130)
    v3 = vision.parse_verdicts(json.dumps({"clips": [{"id": 0, "what_happens": "x", "visual": 6, "hook": 5,
        "menu_or_idle": False, "starts_late": False, "start_earlier": 9}], "ranking": [0]}), 1)
    vision.apply_verdicts([c3], v3, max_len=60)
    assert c3.start == 100


# --- cutscenes and the Claude fallback -------------------------------------------------------------

def _verdict(cid, **kw):
    base = {"id": cid, "what_happens": "x", "visual": 6, "hook": 6, "menu_or_idle": False,
            "starts_late": False, "start_earlier": 0, "cutscene": False, "streamer_reaction": 5, "best_frame": 0}
    base.update(kw)
    return base


def test_cutscene_without_reaction_is_penalised():
    quiet, reacted, gameplay = _cand(0, 30), _cand(40, 70), _cand(80, 110)
    v = vision.parse_verdicts(json.dumps({"clips": [
        _verdict(0, cutscene=True, streamer_reaction=2),
        _verdict(1, cutscene=True, streamer_reaction=9),
        _verdict(2, cutscene=False, streamer_reaction=6)], "ranking": []}), 3)
    vision.apply_verdicts([quiet, reacted, gameplay], v)
    assert quiet.penalty == 2.5 and reacted.penalty == 1.0 and gameplay.penalty == 0
    assert quiet.extra["exclude"] and "exclude" not in reacted.extra and "exclude" not in gameplay.extra
    assert quiet.final_score < reacted.final_score < gameplay.final_score
    assert "cutscene, little reaction" in quiet.vision_note and "strong reaction" in reacted.vision_note


def test_judge_prompt_puts_the_streamer_first():
    from clipfarm.pipeline.judge import SYSTEM_PROMPT

    assert "The streamer is the show" in SYSTEM_PROMPT and "cutscene" in SYSTEM_PROMPT
    assert "endings" in SYSTEM_PROMPT and "spoilers" in SYSTEM_PROMPT
    assert "streamer_reaction" in vision.SYSTEM_PROMPT


def test_claude_blocks_and_strict_schema():
    blocks = vision.claude_blocks([_cand(10, 40)], [[b"\xff\xd8a", b"\xff\xd8b"]], [], before=[[b"\xff\xd8p"]])
    kinds = [b["type"] for b in blocks]
    assert kinds.count("image") == 3 and kinds[0] == "text"
    assert blocks[[i for i, k in enumerate(kinds) if k == "image"][0]]["source"]["media_type"] == "image/jpeg"
    s = vision.strict_schema(vision.VISION_SCHEMA)
    assert s["additionalProperties"] is False and s["properties"]["clips"]["items"]["additionalProperties"] is False
    assert "additionalProperties" not in vision.VISION_SCHEMA  # original untouched (Gemini gets it as is)


def test_claude_takes_over_when_every_gemini_model_fails(monkeypatch):
    calls = {}

    class Judge:
        def __init__(self):
            def gen(model, contents, config):
                raise RuntimeError("503 UNAVAILABLE")
            self.client = SimpleNamespace(models=SimpleNamespace(generate_content=gen))

        def _gemini_client(self):
            return self.client

        def _gemini_chain(self):
            return ["m1", "m2"]

        def claude_allowed(self):
            return True

        def claude_json(self, system, content, schema):
            calls["images"] = sum(1 for b in content if b["type"] == "image")
            calls["closed"] = schema["additionalProperties"] is False
            return json.dumps({"clips": [_verdict(0, visual=9), _verdict(1, cutscene=True, streamer_reaction=1)],
                               "ranking": [0, 1]})

    monkeypatch.setattr(vision, "sample_frames", lambda video, s, e, n=4: [b"\xff\xd8"] * n)
    cands = _cands(2)
    note = vision.VisionReviewer(SimpleNamespace(gemini_api_key="k", clip_max_seconds=60), Judge()).review("v", cands, [])
    assert note == "AI watched 2 of 2 candidates (Claude fallback)"
    assert calls["closed"] and calls["images"] == 2 * 4 + 2  # second clip starts at 100 s: has 2 "before" frames
    assert cands[0].visual == 9 and cands[1].penalty == 2.5


def test_no_claude_when_cap_reached(monkeypatch):
    class Judge:
        def __init__(self):
            def gen(model, contents, config):
                raise RuntimeError("503 UNAVAILABLE")
            self.client = SimpleNamespace(models=SimpleNamespace(generate_content=gen))

        def _gemini_client(self):
            return self.client

        def _gemini_chain(self):
            return ["m1"]

        def claude_allowed(self):
            return False

        def claude_json(self, *a):
            raise AssertionError("must not be called")

    monkeypatch.setattr(vision, "sample_frames", lambda video, s, e, n=4: [b"\xff\xd8"] * n)
    assert "failed" in vision.VisionReviewer(SimpleNamespace(gemini_api_key="k"), Judge()).review("v", _cands(2), [])


def test_best_frame_sets_the_cover():
    c = _cand(100, 120)
    c.extra["frame_times"] = [102.5, 107.5, 112.5, 117.5]
    v = vision.parse_verdicts(json.dumps({"clips": [_verdict(0, best_frame=2)], "ranking": [0]}), 1)
    vision.apply_verdicts([c], v)
    assert c.cover_at == 112.5
    c2 = _cand(100, 120)
    c2.extra["frame_times"] = [102.5]
    vision.apply_verdicts([c2], vision.parse_verdicts(json.dumps({"clips": [_verdict(0, best_frame=9)], "ranking": []}), 1))
    assert c2.cover_at is None  # out of range: ignored
