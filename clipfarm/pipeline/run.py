"""The whole pipeline for one link. Shared by the worker (Telegram) and the CLI."""

from __future__ import annotations

import hashlib
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..db import Database
from ..media import extract_audio, probe, scene_cuts
from ..storage import R2, clip_key
from . import captions, download, layout as layout_mod, render, select, signals, transcribe, twitch, vision
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


@dataclass
class TwitchSignals:
    clips: list[twitch.ViewerClip]
    chat: twitch.ChatData | None
    notes: list[str]  # what worked / what didn't, shown in the job status


def fetch_twitch_signals(cfg, source: download.Source, duration: float) -> TwitchSignals | None:
    """Viewer clips + chat replay for a Twitch VOD. Each part fails independently:
    a missing signal makes picks weaker, never fails the job."""
    vid = twitch.vod_id(source.url, source.video_id, source.extractor)
    if not vid:
        return None
    out = TwitchSignals(clips=[], chat=None, notes=[])
    if cfg.twitch_client_id and cfg.twitch_client_secret:
        try:
            out.clips = twitch.Helix(cfg.twitch_client_id, cfg.twitch_client_secret).vod_clips(vid)
            out.notes.append(f"{len(out.clips)} viewer clips")
        except Exception as exc:
            log.warning("Twitch viewer clips unavailable: %s", exc)
            out.notes.append("viewer clips failed")
    else:
        out.notes.append("viewer clips off (no TWITCH_CLIENT_ID)")
    if cfg.twitch_chat:
        try:
            out.chat = twitch.ChatReplay().fetch(vid, duration)
            out.notes.append(f"{out.chat.total} chat messages")
        except Exception as exc:
            log.warning("Twitch chat replay unavailable: %s", exc)
            out.notes.append("chat replay failed")
    return out


def audience_note(tw: TwitchSignals | None, start: float, end: float) -> str:
    """How the audience reacted inside a clip, in words, for the vision review."""
    if tw is None:
        return ""
    notes = []
    if tw.chat is not None:
        # The busiest 12 s of chat within the clip (chat time = video time + delay).
        a, b = int(start) + twitch.CHAT_DELAY_SECONDS, int(end) + twitch.CHAT_DELAY_SECONDS
        window = tw.chat.rate[a:b]
        if window.size:
            peak = a + int(window.argmax())
            s = tw.chat.summary(max(a, peak - 4))
            if s:
                notes.append(s)
    for c in sorted(twitch.clips_near(tw.clips, start, end), key=lambda c: c.views, reverse=True)[:2]:
        notes.append(f'viewers clipped this: "{c.title}" ({c.views} views)')
    return "; ".join(notes)


def annotate_zones(zones: list[signals.HotZone], tw: TwitchSignals | None) -> None:
    """Attach what chat said and which viewer clips cover each hot zone, for the judge."""
    if tw is None:
        return
    for z in zones:
        if tw.chat is not None:
            summary = tw.chat.summary(z.t + twitch.CHAT_DELAY_SECONDS - 2)
            if summary:
                z.notes.append(summary)
        near = sorted(twitch.clips_near(tw.clips, z.t - 5, z.t + 5), key=lambda c: c.views, reverse=True)
        for c in near[:2]:
            z.notes.append(f'viewer clip "{c.title}" ({c.views} views)')


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
    attempt: int = 1,
) -> list[RenderedClip]:
    workdir = work_dir_for(cfg, url)
    # A re-run (after a worker restart) gets its own folder: earlier clips may already be approved.
    outdir = cfg.clips_dir / f"job-{job_id}" / f"try-{attempt}"
    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        result = _process(url, job_id, cfg, workdir, outdir, transcriber, db, on_status, on_clip, on_source)
    except BaseException:
        # Keep the (possibly multi-GB) download: sending the same link again skips re-downloading.
        # Old leftovers are removed by the regular cleanup.
        log.info("Job failed; keeping %s so a retry of this link reuses the download", workdir)
        raise
    if not keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    return result


def work_dir_for(cfg, url: str) -> Path:
    """Scratch folder per source URL (not per job), so retries find the earlier download."""
    return cfg.work_dir / ("src-" + hashlib.sha1(url.strip().encode()).hexdigest()[:12])


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

    # Twitch chat/clips download over the network while Whisper uses the CPU.
    with ThreadPoolExecutor(1) as pool:
        tw_future = pool.submit(fetch_twitch_signals, cfg, source, duration)

        # 2. Transcript ---------------------------------------------------------
        on_status("transcribing", "using site subtitles" if source.subtitles_path else "running Whisper")
        segments: list[transcribe.Segment] = []
        if source.subtitles_path:
            segments = transcribe.parse_json3(source.subtitles_path)
        if len(segments) < 10 and audio is not None:
            segments = transcriber.transcribe(audio)
        if not segments:
            raise PipelineError("No speech found; there's nothing for the judge to read.")

        if not tw_future.done():
            on_status("transcribing", "waiting for Twitch chat")
        tw = tw_future.result()

    # 3. Signals + judge --------------------------------------------------------
    extra: dict = {}
    if tw is not None:
        if tw.clips:
            extra[signals.VIEWER_CLIPS] = twitch.viewer_clip_signal(tw.clips, int(duration) + 1)
        if tw.chat is not None and tw.chat.total > 0:
            extra[signals.CHAT] = twitch.chat_signal(tw.chat)
    loud = signals.loudness_per_second(audio) if audio is not None else None
    sig = signals.build_signals(duration, loud, source.heatmap, extra)
    on_status("analyzing", "signals: " + sig.describe() + (f"; {', '.join(tw.notes)}" if tw else ""))
    zones = signals.hot_zones(sig)
    annotate_zones(zones, tw)
    judge = Judge(cfg, db)
    candidates = judge.pick(
        title=source.title,
        uploader=source.uploader,
        duration=duration,
        transcript=transcribe.format_for_llm(segments),
        zones=zones,
        viewer_clips=tw.clips if tw else None,
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
        c.extra["audience"] = audience_note(tw, start, end)
        snapped.append(c)

    # Second look: Gemini watches frames of the best non-overlapping candidates and ranks them.
    shortlist = select.choose(snapped, max(cfg.clips_per_video, cfg.vision_candidates))
    if cfg.vision_review and cfg.gemini_api_key and len(shortlist) > 1:
        on_status("analyzing", f"AI is watching the top {len(shortlist)} candidates")
        note = vision.VisionReviewer(cfg, judge).review(source.video_path, shortlist, segments)
        log.info("Vision review: %s", note)
    chosen = select.choose(shortlist, cfg.clips_per_video)
    # Present in story order; scores are shown on each clip.
    chosen.sort(key=lambda c: c.start)
    if not chosen:
        raise PipelineError("The judge found no usable clips in this video.")

    # 4. Render -------------------------------------------------------------------
    presets = layout_mod.load_presets(cfg.presets_path)
    stream_cam = None
    cam_searched = False
    if vinfo.width > vinfo.height and (source.uploader or "").lower() not in presets and cfg.face_model_path.exists():
        on_status("rendering", "finding the streamer's webcam")
        stream_cam = layout_mod.find_stream_webcam(
            source.video_path, duration, vinfo.width, vinfo.height, cfg.face_model_path)
        cam_searched = True
        if stream_cam:
            log.info("Webcam found at x=%.0f y=%.0f (face %.0fx%.0f px)", *stream_cam)
        else:
            log.info("No fixed webcam in this stream: gameplay clips use the full picture")
    r2 = R2(cfg) if cfg.r2_enabled else None
    rendered: list[RenderedClip] = []
    for i, c in enumerate(chosen, start=1):
        on_status("rendering", f"{i}/{len(chosen)}")
        lay = layout_mod.choose_layout(
            source.video_path, c.start, c.end, vinfo.width, vinfo.height,
            cfg.face_model_path, source.uploader, presets,
            stream_cam=stream_cam, cam_searched=cam_searched,
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
            key = clip_key(job_id, i, attempt)
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
