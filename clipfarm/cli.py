"""Command line, for testing without Telegram.

    python -m clipfarm.cli process <url> [--clips N] [--keep-work]
    python -m clipfarm.cli cleanup
    python -m clipfarm.cli section <url> 6:00:00-6:01:00   (download test, nothing else)
    python -m clipfarm.cli youtube-login                    (once, to let ClipFarm upload Shorts)
"""

from __future__ import annotations

import argparse
import sys
import dataclasses
import logging

from . import db as dbm
from .config import load_config
from .logsetup import setup_logging
from .pipeline.run import process
from .pipeline.transcribe import Transcriber
from .storage import R2, cleanup_local


def cmd_process(args) -> None:
    cfg = load_config()
    if args.clips:
        cfg = dataclasses.replace(cfg, clips_per_video=args.clips)
    db = dbm.Database(cfg.db_path)
    job_id = db.add_job(args.url)
    db.update_job(job_id, status="downloading")
    transcriber = Transcriber(cfg.whisper_model, cfg.whisper_compute_type, cfg.whisper_threads)

    def on_status(status: str, detail: str) -> None:
        db.update_job(job_id, status=status)
        print(f"[{status}] {detail}".rstrip())

    def on_clip(rc) -> None:
        c = rc.candidate
        print(
            f"  clip {rc.idx}/{rc.total}  {c.start:7.1f}-{c.end:7.1f}s  score {c.final_score:4.1f}  "
            f"{rc.layout:8s} {c.title}\n"
            + (f"      saw: {c.vision_note}\n" if c.vision_note else "")
            + f"      -> {rc.final_path}"
        )

    try:
        clips = process(
            args.url, job_id, cfg, transcriber=transcriber, db=db,
            on_status=on_status, on_clip=on_clip, keep_workdir=args.keep_work,
            on_source=lambda src: db.update_job(job_id, source_title=src.title, uploader=src.uploader,
                                                duration=src.duration),
        )
    except Exception as exc:
        db.update_job(job_id, status=dbm.FAILED, error=str(exc))
        raise
    db.update_job(job_id, status=dbm.DONE)
    print(f"Done: {len(clips)} clips in {cfg.clips_dir / f'job-{job_id}'}")


def cmd_cleanup(_args) -> None:
    cfg = load_config()
    print("local folders removed:", cleanup_local(cfg.clips_dir, cfg.local_retention_days, cfg.work_dir))
    if cfg.r2_enabled:
        print("R2 objects deleted:", R2(cfg).cleanup(cfg.r2_retention_days))


def cmd_section(args) -> None:
    """Download only a short range and report what came back: a quick check of long-stream parts."""
    import shutil
    import tempfile
    import time
    from pathlib import Path

    from .longform import hms, parse_range
    from .media import probe
    from .pipeline import download

    cfg = load_config()
    rng = parse_range(" " + args.range, args.url, default_len=60)
    if rng is None:
        sys.exit("Give a range like 6:00:00-6:01:00")
    work = Path(tempfile.mkdtemp(prefix="section-", dir=cfg.data_dir))
    t0 = time.monotonic()
    try:
        info = download.probe_info(args.url, cfg.ytdlp_cookies)
        fmt = download.pick_hls_video(info)
        print(f"video variant: {fmt.get('format_id') if fmt else 'none (not HLS)'}"
              + (f" · {fmt.get('vcodec')} · {fmt.get('height')}p" if fmt else ""))
        src = download.download(args.url, work, cfg.ytdlp_cookies, rng)
        v = probe(src.video_path)
        print(f"OK in {time.monotonic() - t0:.0f}s: asked {hms(rng[0])}-{hms(rng[1])}, file starts at "
              f"{hms(src.section_start or rng[0])}, {v.duration:.0f}s long, {v.width}x{v.height}, "
              f"audio {'yes' if v.has_audio else 'NO'}, {src.video_path.stat().st_size / 1e6:.0f} MB")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def cmd_youtube_login(_args) -> None:
    import httpx

    from .publish import PublishError
    from .publish import youtube as yt

    cfg = load_config()
    try:
        client = yt.load_client(cfg.youtube_client_path)
    except PublishError as exc:
        sys.exit(f"{exc}\nCopy the OAuth client JSON (Desktop app) from Google Cloud to {cfg.youtube_client_path}")
    login = yt.new_login()
    print("1. Open this link in your browser (on any computer) and sign in with the Google account")
    print("   that owns the YouTube channel. If Google asks which channel, pick the one to post to.")
    print("   'Google hasn't verified this app' is expected: Advanced > Go to ... (unsafe) > Continue.\n")
    print(yt.auth_url(client[0], login))
    print("\n2. After you allow access, the browser ends on a page that doesn't load (127.0.0.1:8765).")
    print("   That's expected. Copy the WHOLE address from the address bar and paste it here.\n")
    for tries in range(3):
        try:
            code = yt.code_from_paste(input("Address: "), login["state"])
            break
        except PublishError as exc:
            print(f"  {exc}")
    else:
        sys.exit("Run the command again when you have the address.")
    try:
        with httpx.Client(timeout=60) as http:
            token = yt.exchange_code(http, client, code, login["verifier"])
            yt.save_token(cfg.youtube_token_path, token)
            channel = yt.YouTube(cfg, http=http).channel()
    except PublishError as exc:
        sys.exit(f"Login failed: {exc}")
    print(f"\nLogged in. Shorts will go to the channel: {channel['title']}")
    print("Approved clips are now booked for YouTube too (no restart needed).")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # `> run.log` would otherwise buffer status lines
    setup_logging()
    parser = argparse.ArgumentParser(prog="clipfarm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process", help="Run the full pipeline on one link")
    p.add_argument("url")
    p.add_argument("--clips", type=int, help="Override CLIPS_PER_VIDEO")
    p.add_argument("--keep-work", action="store_true", help="Keep the downloaded source and audio")
    p.set_defaults(func=cmd_process)

    s = sub.add_parser("section", help="Download test: only a short range of a long stream")
    s.add_argument("url")
    s.add_argument("range", help="e.g. 6:00:00-6:01:00")
    s.set_defaults(func=cmd_section)

    y = sub.add_parser("youtube-login", help="Let ClipFarm upload to your YouTube channel (once)")
    y.set_defaults(func=cmd_youtube_login)

    c = sub.add_parser("cleanup", help="Delete clips older than the retention window")
    c.set_defaults(func=cmd_cleanup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
