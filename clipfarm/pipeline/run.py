"""The whole pipeline for one link. Shared by the worker (Telegram) and the CLI."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..db import Database
from ..media import extract_audio, probe, scene_cuts
from ..storage import R2, clip_key
from . import captions, download, layout as layout_mod, render, select, signals, transcribe
from .judge import Candidate, Judge

log = logging.getLogger(__name__)

StatusFn = Callable[[str, str], None]


@dataclass
class RenderedClip:
    idx: int
    total: int
    candidate: Candidate
    layout: str
    final_path: Path
    preview_path: Path
    r2_key: str | None
    public_url: str | None


class PipelineError(RuntimeError):
    pass


def process(
    url: str,
    job_id: int,
    cfg,
    *,
    transcriber: transcribe.Transcriber,
    db: Database | None = None,
    on_status: StatusFn = lambda status, detail: None,
    on_clip: Callable[[RenderedClip], None] = lambda clip: None,
    on_source: Callable[[download.Source], None] = lambda src: None,
    keep_workdir: bool = False,
) -> list[RenderedClip]:
    workdir = cfg.work_dir / f"job-{job_id}"
    outdir = cfg.clips_dir / f"job-{job_id}"
    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        return _process(url, job_id, cfg, workdir, outdir, transcriber, db, on_status, on_clip, on_source)
    finally:
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def _process(url, job_id, cfg, workdir, outdir, transcriber, db, on_status, on_clip, on_source):
    # 1. Download -------------------------------------------------------------
    on_status("downloading", "")
    meta = download.probe_info(url, cfg.ytdlp_cookies)
    if meta.get("is_live"):
        raise PipelineError("That stream is still live. Send the link again once the VOD is available.")
    if meta.get("duration") and meta["duration"] > cfg.max_source_hours * 3600:
        raise PipelineError(
            f"Video is {meta['duration'] / 3600:.1f} h; the limit is {cfg.max_source_hours:g} h (MAX_SOURCE_HOURS)."
        )
    source = download.download(url, workdir, cfg.ytdlp_cookies)
    vinfo = probe(source.video_path)
    duration = vinfo.duration or source.duration
    source.duration = duration
    on_source(source)

    audio = extract_audio(source.video_path, workdir / "audio.wav") if vinfo.has_audio else None

    # 2. Transcript -------------------------------------------------------------
    on_status("transcribing", "using site subtitles" if source.subtitles_path else "running Whisper")
    segments: list[transcribe.Segment] = []
    if source.subtitles_path:
        segments = transcribe.parse_json3(source.subtitles_path)
    if len(segments) < 10 and audio is not None:
        segments = transcriber.transcribe(audio)
    if not segments:
        raise PipelineError("No speech found; there's nothing for the judge to read.")

    # 3. Signals + judge --------------------------------------------------------
    on_status("analyzing", "")
    loud = signals.loudness_per_second(audio) if audio is not None else None
    sig = signals.build_signals(duration, loud, source.heatmap)
    zones = signals.hot_zones(sig)
    candidates = Judge(cfg, db).pick(
        title=source.title,
        uploader=source.uploader,
        duration=duration,
        transcript=transcribe.format_for_llm(segments),
        zones=zones,
    )

    snapped: list[Candidate] = []
    for c in candidates:
        cuts = scene_cuts(source.video_path, c.start - 3, c.start + 3) + scene_cuts(
            source.video_path, c.end - 3, c.end + 3
        )
        start, end = select.snap_boundaries(
            c.start, c.end, segments, cuts, cfg.clip_min_seconds, cfg.clip_max_seconds, duration
        )
        c = select.with_bounds(c, start, end)
        c.signal_score = sig.strength(start, end)
        snapped.append(c)
    chosen = select.choose(snapped, cfg.clips_per_video)
    # Present in story order; scores are shown on each clip.
    chosen.sort(key=lambda c: c.start)
    if not chosen:
        raise PipelineError("The judge found no usable clips in this video.")

    # 4. Render -------------------------------------------------------------------
    presets = layout_mod.load_presets(cfg.presets_path)
    r2 = R2(cfg) if cfg.r2_enabled else None
    rendered: list[RenderedClip] = []
    for i, c in enumerate(chosen, start=1):
        on_status("rendering", f"{i}/{len(chosen)}")
        lay = layout_mod.choose_layout(
            source.video_path, c.start, c.end, vinfo.width, vinfo.height,
            cfg.face_model_path, source.uploader, presets,
        )
        ass_path = None
        if audio is not None:
            clip_audio = extract_audio(source.video_path, workdir / f"clip-{i:02d}.wav", c.start, c.end)
            words = [w for s in transcriber.transcribe(clip_audio, words=True) for w in s.words]
            if words:
                ass_path = captions.write_ass(
                    words, workdir / f"clip-{i:02d}.ass", margin_v=render.caption_margin(lay)
                )
        final = render.render_clip(
            source.video_path, c.start, c.end, lay, ass_path, outdir / f"clip-{i:02d}.mp4", vinfo.has_audio
        )
        preview = render.render_preview(final, outdir / f"clip-{i:02d}.preview.mp4")

        key = url_ = None
        if r2 is not None:
            key = clip_key(job_id, i)
            try:
                url_ = r2.upload(final, key, "video/mp4")
                r2.upload_json(
                    {
                        "source_url": url, "source_title": source.title, "start": c.start, "end": c.end,
                        "title": c.title, "hashtags": c.hashtags, "reason": c.reason,
                        "llm_score": c.llm_score, "signal_score": c.signal_score, "judge_model": c.model,
                    },
                    key.replace(".mp4", ".json"),
                )
            except Exception as exc:  # a storage hiccup shouldn't lose a rendered clip
                log.warning("R2 upload failed for %s: %s", key, exc)
                key = url_ = None

        clip = RenderedClip(
            idx=i, total=len(chosen), candidate=c, layout=lay.kind,
            final_path=final, preview_path=preview, r2_key=key, public_url=url_,
        )
        rendered.append(clip)
        on_clip(clip)
    return rendered
