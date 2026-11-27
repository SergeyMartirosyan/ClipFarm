"""End-to-end run of the pipeline on a synthetic video.

Real: ffmpeg cutting/layout/captions/preview, loudness signals, scene cuts, snapping, DB.
Stubbed: the network (yt-dlp), the LLM judge, and Whisper.
"""

import dataclasses
import json
import shutil
import subprocess

import pytest

from clipfarm import db as dbm
from clipfarm.config import load_config
from clipfarm.media import probe
from clipfarm.pipeline import download, run
from clipfarm.pipeline.judge import Candidate, Judge
from clipfarm.pipeline.transcribe import Segment, Word

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def source_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("src") / "source.mp4"
    # 90 s 1080p test pattern; quiet tone with a loud burst at 40-43 s.
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30:duration=90",
            "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000:duration=90",
            "-filter_complex", "[1:a]volume='if(between(t,40,43),1.0,0.05)':eval=frame[a]",
            "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
            str(path),
        ],
        check=True,
    )
    return path


class StubTranscriber:
    def transcribe(self, audio_path, words=False):
        segs = []
        for t in range(0, 90, 5):
            seg = Segment(float(t), t + 4.5, f"sentence number {t}")
            if words:
                seg.words = [Word(t + i * 0.5, t + i * 0.5 + 0.4, w) for i, w in enumerate("this is a test".split())]
            segs.append(seg)
        return segs


def test_full_pipeline(tmp_path, monkeypatch, source_video):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    presets = tmp_path / "presets.json"
    presets.write_text(json.dumps({"teststreamer": {"cam": [0.0, 0.6, 0.25, 0.4], "game": [0.2, 0, 0.6, 1.0]}}))
    monkeypatch.setenv("PRESETS_PATH", str(presets))
    cfg = dataclasses.replace(load_config(env_file=str(tmp_path / "none.env")), clips_per_video=2)
    db = dbm.Database(cfg.db_path)
    job_id = db.add_job("https://example.com/v")

    def fake_download(url, workdir, cookies="", section=None):
        workdir.mkdir(parents=True, exist_ok=True)
        dst = workdir / "source.mp4"
        shutil.copy(source_video, dst)
        return download.Source(url=url, video_path=dst, title="Test", uploader="TestStreamer",
                               duration=90, extractor="test")

    captured = {}

    def fake_pick(self, *, title, uploader, duration, transcript, zones, viewer_clips=None):
        captured["zones"] = zones
        captured["transcript"] = transcript
        loud = Candidate(36.5, 61.0, "The loud part", ["test"], 9, 8, 8, "burst")
        loud.texts = [{"overlay": "He did NOT see that coming", "caption": "No way 💀"},
                      {"overlay": "POV: the loudest second of the stream", "caption": "Ears 😭"}]
        loud.overlay, loud.caption = loud.texts[0]["overlay"], loud.texts[0]["caption"]
        return [
            loud,
            Candidate(5.2, 30.0, "The quiet part", ["test"], 5, 5, 5, "calm"),
            Candidate(40.0, 50.0, "Overlaps the first", [], 3, 3, 3, ""),
        ]

    monkeypatch.setattr(download, "probe_info", lambda url, cookies="": {"duration": 90})
    monkeypatch.setattr(download, "download", fake_download)
    monkeypatch.setattr(Judge, "pick", fake_pick)

    statuses, clips = [], []
    result = run.process(
        "https://example.com/v", job_id, cfg, transcriber=StubTranscriber(), db=db,
        on_status=lambda s, d: statuses.append(s), on_clip=clips.append,
    )

    assert statuses[:4] == ["downloading", "transcribing", "analyzing", "rendering"]
    # The loudness burst must show up as a hot zone for the judge.
    assert any(38 <= z.t <= 45 for z in captured["zones"])
    assert "[0.0] sentence number 0" in captured["transcript"]

    assert len(result) == 2 and len(clips) == 2
    first, second = sorted(result, key=lambda c: c.candidate.start)
    assert second.candidate.title == "The loud part"
    assert second.layout == "facecam"  # from the channel preset
    for rc in result:
        info = probe(rc.final_path)
        assert (info.width, info.height) == (1080, 1920)
        assert 20 <= info.duration <= 61
        assert info.has_audio
        assert rc.preview_path.stat().st_size < rc.final_path.stat().st_size
        assert rc.candidate.signal_score >= 0
    assert second.candidate.signal_score > first.candidate.signal_score
    # Two hook texts: a hook-free master is kept for swapping later; one text: none is made.
    assert second.base_path and probe(second.base_path).height == 1920 and first.base_path is None
    # Work directory is cleaned up; final clips are kept.
    assert not (cfg.work_dir / f"job-{job_id}").exists()
    assert (cfg.clips_dir / f"job-{job_id}" / "try-1" / "clip-01.mp4").exists()


def test_blur_layout_render(tmp_path, source_video):
    from clipfarm.pipeline import captions, render
    from clipfarm.pipeline.layout import Layout, choose_layout

    lay = choose_layout(source_video, 10, 20, 1920, 1080, tmp_path / "missing.onnx")
    assert lay.kind == "blur"
    ass = captions.write_ass([Word(0.1, 0.5, "hello"), Word(0.5, 1.0, "world")], tmp_path / "c.ass")
    out = render.render_clip(source_video, 10, 20, lay, ass, tmp_path / "blur.mp4")
    info = probe(out)
    assert (info.width, info.height) == (1080, 1920) and 9.5 <= info.duration <= 10.5
    out2 = render.render_clip(source_video, 10, 15, Layout("face", main=_box()), None, tmp_path / "face.mp4")
    assert probe(out2).height == 1920


def _box():
    from clipfarm.pipeline.layout import Box

    return Box(600, 0, 608, 1080)
