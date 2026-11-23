import json

import numpy as np
import pytest

from clipfarm import db as dbm
from clipfarm import messages
from clipfarm.pipeline import captions, layout, select, signals, transcribe
from clipfarm.pipeline.judge import CLIP_SCHEMA, Candidate, _gemini_schema, build_user_prompt, parse_candidates
from clipfarm.pipeline.transcribe import Segment, Word


# --- transcript ------------------------------------------------------------------

def test_parse_json3_clamps_overlaps_and_skips_appends(tmp_path):
    data = {
        "events": [
            {"tStartMs": 0, "dDurationMs": 5000, "segs": [{"utf8": "hello"}, {"utf8": " world"}]},
            {"tStartMs": 2000, "dDurationMs": 10, "aAppend": 1, "segs": [{"utf8": "\n"}]},
            {"tStartMs": 3000, "dDurationMs": 4000, "segs": [{"utf8": "second  line"}]},
            {"tStartMs": 8000, "dDurationMs": 1000},
        ]
    }
    p = tmp_path / "subs.en.json3"
    p.write_text(json.dumps(data))
    segs = transcribe.parse_json3(p)
    assert [s.text for s in segs] == ["hello world", "second line"]
    assert segs[0].end == pytest.approx(3.0)  # clamped to the next line's start
    assert segs[1].end == pytest.approx(7.0)


def test_format_for_llm():
    out = transcribe.format_for_llm([Segment(1.25, 2, "hi"), Segment(65.0, 66, "there")])
    assert out == "[1.2] hi\n[65.0] there" or out == "[1.3] hi\n[65.0] there"


# --- signals -----------------------------------------------------------------------

def test_loudness_spike_becomes_hot_zone():
    db = np.full(600, -30.0, dtype=np.float32)
    db += np.random.default_rng(0).normal(0, 0.5, 600).astype(np.float32)
    db[300:304] = -8.0  # a shout
    sig = signals.build_signals(600, db, None)
    zones = signals.hot_zones(sig, limit=3)
    assert zones and abs(zones[0].t - 301) <= 3
    assert "loudness spike" in zones[0].reasons
    assert sig.strength(295, 310) == pytest.approx(1.0)


def test_heatmap_score_stretches_contrast():
    hm = [{"start_time": i * 10, "end_time": (i + 1) * 10, "value": 0.3} for i in range(10)]
    hm[4]["value"] = 1.0
    s = signals.heatmap_score(hm, 100)
    assert s[45] == pytest.approx(1.0) and s[5] == pytest.approx(0.0)


def test_no_signals_is_all_zero():
    sig = signals.build_signals(120, None, None)
    assert sig.combined.shape == (120,) and sig.combined.max() == 0
    assert signals.hot_zones(sig) == []


# --- judge parsing ---------------------------------------------------------------------

def test_parse_candidates_validates_and_clamps():
    raw = json.dumps({"clips": [
        {"start": 10, "end": 40, "title": "Good", "hashtags": ["#Gaming", "fail"], "hook": 12,
         "standalone": 7, "payoff": "8", "reason": "funny"},
        {"start": 50, "end": 40, "title": "backwards", "hashtags": [], "hook": 1, "standalone": 1,
         "payoff": 1, "reason": ""},
        {"start": 90, "end": 500, "title": "past end", "hashtags": [], "hook": 1, "standalone": 1,
         "payoff": 1, "reason": ""},
    ]})
    cands = parse_candidates(raw, duration=100, model="m")
    assert len(cands) == 1
    c = cands[0]
    assert c.hook == 10 and c.payoff == 8 and c.hashtags == ["gaming", "fail"]


def test_gemini_schema_drops_additional_properties():
    s = _gemini_schema(CLIP_SCHEMA)
    assert "additionalProperties" not in json.dumps(s)
    assert "additionalProperties" in json.dumps(CLIP_SCHEMA)  # original untouched


def test_prompt_contains_zones_and_taste():
    p = build_user_prompt(
        title="T", uploader="U", duration=100, transcript="[1.0] hi",
        zones=[signals.HotZone(50, 0.9, ["replay peak"])], want=5, min_len=20, max_len=60,
        approved_examples=['"A" (x)'], skipped_examples=['"B" (y)'],
    )
    assert "50s  0.90  replay peak" in p and "APPROVED" in p and "SKIPPED" in p and "[1.0] hi" in p


# --- boundary snapping + choosing ---------------------------------------------------------

SEGS = [Segment(float(t), float(t) + 4.5, f"s{t}") for t in range(0, 300, 5)]


def test_snap_moves_to_sentence_boundaries():
    s, e = select.snap_boundaries(11.5, 42.0, SEGS, [], 20, 60, 300)
    assert s == 10.0 and e == 44.5


def test_snap_prefers_nearby_scene_cut():
    s, e = select.snap_boundaries(11.5, 42.0, SEGS, [10.4], 20, 60, 300)
    assert s == 10.4


def test_snap_enforces_length():
    s, e = select.snap_boundaries(10, 200, SEGS, [], 20, 60, 300)
    assert e - s <= 60
    s, e = select.snap_boundaries(10, 15, SEGS, [], 20, 60, 300)
    assert e - s >= 20


def test_snap_near_video_end():
    s, e = select.snap_boundaries(290, 299, SEGS, [], 20, 60, 300)
    assert e <= 300 and e - s >= 20


def _cand(start, end, hook):
    return Candidate(start, end, "t", [], hook, 5, 5, "")


def test_choose_drops_overlaps_keeps_best():
    a, b, c = _cand(0, 30, 9), _cand(10, 40, 5), _cand(50, 80, 7)
    chosen = select.choose([b, a, c], limit=5)
    assert chosen == [a, c]


# --- layout -------------------------------------------------------------------------------

def test_layout_facecam_for_small_steady_face():
    faces = [[(100, 800, 120, 140)] for _ in range(10)]
    lay = layout.decide(faces, 1920, 1080)
    assert lay.kind == "facecam"
    cam, game = lay.cam, lay.main
    assert 0 <= cam.x and cam.x + cam.w <= 1920 and cam.y + cam.h <= 1080
    assert abs(cam.w / cam.h - 1080 / 768) < 0.05
    assert game.h == 1080 and abs(game.w / game.h - 1080 / 1152) < 0.01
    assert all(v % 2 == 0 for v in (cam.w, cam.h, game.w, game.h))


def test_layout_face_follow_for_big_face():
    faces = [[(1200, 200, 500, 600)] for _ in range(10)]
    lay = layout.decide(faces, 1920, 1080)
    assert lay.kind == "face"
    assert lay.main.h == 1080 and lay.main.x + lay.main.w <= 1920
    assert abs(lay.main.w / 1080 - 9 / 16) < 0.01


def test_layout_blur_without_faces_and_vertical_source():
    assert layout.decide([[] for _ in range(10)], 1920, 1080).kind == "blur"
    assert layout.decide([], 1080, 1920).kind == "vertical"


# --- captions -------------------------------------------------------------------------------

def test_caption_grouping_and_ass():
    words = [Word(0.0, 0.3, "So"), Word(0.3, 0.6, "this"), Word(0.6, 0.9, "is"),
             Word(0.9, 1.2, "crazy."), Word(2.5, 2.8, "Wait{x}")]
    groups = captions.group_words(words)
    assert [len(g) for g in groups] == [3, 1, 1]
    ass = captions.build_ass(words)
    assert ass.count("Dialogue:") == 5
    assert "WAIT(X)" in ass and "{x}" not in ass
    assert "0:00:00.00" in ass


# --- db + messages ----------------------------------------------------------------------------

def test_db_job_lifecycle_and_messages(tmp_path):
    d = dbm.Database(tmp_path / "t.db")
    j1, j2 = d.add_job("https://a"), d.add_job("https://b")
    assert d.queue_position(j2) == 2
    claimed = d.claim_next_job()
    assert claimed["id"] == j1 and claimed["status"] == "downloading"
    d.update_job(j1, status="rendering")
    assert d.requeue_stuck_jobs() == 1
    cid = d.add_clip(job_id=j1, idx=1, total=3, start=10, end=40, title="Hi", hashtags="a b",
                     final_score=7.5, layout="facecam", judge_model="m")
    clip = d.get_clip(cid)
    assert "Clip 1/3" in messages.clip_caption(clip) and "#a #b" in messages.clip_caption(clip)
    assert messages.clip_buttons(clip)[0][0][1] == f"ok:{cid}"
    d.set_clip_decision(cid, dbm.APPROVED)
    clip = d.get_clip(cid)
    assert "Approved" in messages.clip_caption(clip)
    assert messages.clip_buttons(clip) == [[("↩️ Undo", f"undo:{cid}")]]
    assert d.recent_decisions(dbm.APPROVED)[0]["id"] == cid
    d.record_llm_usage("anthropic", "m", 1000, 100, 0.01)
    assert d.spend_this_month("anthropic") == pytest.approx(0.01)
    d.kv_set("k", "v")
    assert d.kv_get("k") == "v"
    d.kv_set("k", None)
    assert d.kv_get("k") is None


# --- judge fallback ------------------------------------------------------------------------------

def _judge(tmp_path, monkeypatch, cap=2.0):
    import dataclasses

    from clipfarm.config import load_config
    from clipfarm.pipeline import judge as judge_mod

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    cfg = dataclasses.replace(
        load_config(env_file=str(tmp_path / "none.env")),
        gemini_api_key="g", anthropic_api_key="a", fallback_monthly_cap_usd=cap,
    )
    monkeypatch.setattr(judge_mod.time, "sleep", lambda s: None)
    db = dbm.Database(tmp_path / "j.db")
    return judge_mod, judge_mod.Judge(cfg, db), db


GOOD = json.dumps({"clips": [{"start": 1, "end": 30, "title": "x", "hashtags": [], "hook": 5,
                              "standalone": 5, "payoff": 5, "reason": ""}]})
PICK = dict(title="t", uploader="u", duration=100, transcript="[1.0] hi", zones=[])


def test_judge_uses_gemini_when_it_works(tmp_path, monkeypatch):
    jm, j, _ = _judge(tmp_path, monkeypatch)
    monkeypatch.setattr(jm.Judge, "_call_gemini", lambda self, p: GOOD)
    monkeypatch.setattr(jm.Judge, "_call_claude", lambda self, p: pytest.fail("fallback used"))
    assert j.pick(**PICK)[0].model == j.cfg.gemini_model


def test_judge_falls_back_to_claude(tmp_path, monkeypatch):
    jm, j, _ = _judge(tmp_path, monkeypatch)
    calls = []

    def boom(self, p):
        calls.append(1)
        raise RuntimeError("429 quota")

    monkeypatch.setattr(jm.Judge, "_call_gemini", boom)
    monkeypatch.setattr(jm.Judge, "_call_claude", lambda self, p: GOOD)
    assert j.pick(**PICK)[0].model == j.cfg.fallback_model
    assert len(calls) == 3


def test_judge_respects_fallback_cap(tmp_path, monkeypatch):
    jm, j, db = _judge(tmp_path, monkeypatch, cap=0.5)
    db.record_llm_usage("anthropic", "m", 0, 0, 0.6)
    monkeypatch.setattr(jm.Judge, "_call_gemini", lambda self, p: "not json")
    monkeypatch.setattr(jm.Judge, "_call_claude", lambda self, p: pytest.fail("cap ignored"))
    with pytest.raises(jm.JudgeBudgetExceeded):
        j.pick(**PICK)


# --- config + gemini model picking ---------------------------------------------------------------

def test_empty_env_values_fall_back_to_defaults(tmp_path, monkeypatch):
    from clipfarm.config import load_config

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("TELEGRAM_API_BASE", "")
    monkeypatch.setenv("GEMINI_MODEL", "")
    cfg = load_config(env_file=str(tmp_path / "none.env"))
    assert cfg.telegram_api_base == "https://api.telegram.org"
    assert cfg.gemini_model == "gemini-flash-latest"


def test_pick_flash_model_prefers_stable_newest_full_flash():
    from clipfarm.pipeline.judge import pick_flash_model

    names = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.0-flash-preview",
             "gemini-2.5-pro", "gemini-2.5-flash-preview-tts", "gemini-2.5-flash-image", "embedding-001"]
    assert pick_flash_model(names) == "gemini-2.5-flash"
    assert pick_flash_model(["gemini-3.0-flash-preview", "gemini-2.0-flash-lite"]) == "gemini-3.0-flash-preview"
    assert pick_flash_model(["gemini-2.5-pro"]) is None


def test_resolve_gemini_model_falls_back_when_missing():
    from types import SimpleNamespace

    from clipfarm.pipeline import judge as jm

    jm._RESOLVED_MODELS.clear()
    models = [SimpleNamespace(name="models/gemini-2.5-flash", supported_actions=["generateContent"]),
              SimpleNamespace(name="models/gemini-2.5-flash-lite", supported_actions=["generateContent"]),
              SimpleNamespace(name="models/text-embedding-004", supported_actions=["embedContent"])]
    client = SimpleNamespace(models=SimpleNamespace(list=lambda: models))
    assert jm.resolve_gemini_model(client, "gemini-flash-latest") == "gemini-2.5-flash"
    assert jm.resolve_gemini_model(client, "gemini-2.5-flash-lite") == "gemini-2.5-flash-lite"
    jm._RESOLVED_MODELS.clear()
