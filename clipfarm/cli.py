"""Command line, for testing without Telegram.

    python -m clipfarm.cli process <url> [--clips N] [--keep-work]
    python -m clipfarm.cli cleanup
    python -m clipfarm.cli section <url> 6:00:00-6:01:00   (download test, nothing else)
    python -m clipfarm.cli youtube-login                    (once, to let ClipFarm upload Shorts)
    python -m clipfarm.cli reset-posts                      (un-approve everything not yet posted)
    python -m clipfarm.cli retry-failed [--platform instagram]   (post failed ones again, right now)
    python -m clipfarm.cli make-cover <image>               (the cover used for every clip)
    python -m clipfarm.cli tiktok-login                     (once, to let ClipFarm post to TikTok)
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


def cmd_reset_posts(args) -> None:
    """Approved clips back to review and their booked posts cancelled, so ✅ Post books them again
    (on every platform connected now). Their Telegram messages get the buttons back."""
    import time

    from . import messages
    from .notify import Telegram, TelegramError

    cfg = load_config()
    db = dbm.Database(cfg.db_path)
    n = db.clip_counts().get(dbm.APPROVED, 0)
    if not n:
        print("No approved clips waiting to be posted.")
        return
    if not args.yes and input(f"Put {n} approved clips back to review and cancel their posts? [y/N] ").lower() != "y":
        return
    rows = db.reset_approved()
    print(f"Reset {len(rows)} clips.")
    tg = Telegram(cfg.telegram_token, cfg.telegram_api_base) if cfg.telegram_enabled else None
    fixed = 0
    for row in rows:
        chat = row["chat_id"] or cfg.telegram_chat_id or cfg.telegram_owner_id
        if not tg or not row["telegram_message_id"] or not chat:
            continue
        clip = db.get_clip(row["id"])
        try:
            tg.edit_caption(int(chat), clip["telegram_message_id"],
                            messages.clip_caption(clip, db.posts_for_clip(clip["id"]), cfg.post_tz),
                            messages.buttons_as_json(messages.clip_buttons(clip)))
            fixed += 1
        except (TelegramError, OSError) as exc:
            print(f"  clip {clip['id']}: couldn't update its Telegram message ({exc})")
        time.sleep(1)  # stay under Telegram's edit rate limit
    if tg:
        print(f"Updated {fixed} Telegram messages: tap ✅ Post on the ones you want.")


def cmd_retry_failed(args) -> None:
    """Post every failed post again now, one by one, printing the outcome (and the group is told)."""
    import time

    from .poster import Poster

    poster = Poster()
    rows = poster.db.failed_posts(args.platform)
    if not rows:
        print("No failed posts" + (f" for {args.platform}" if args.platform else "") + ".")
        return
    if args.first:
        rows = rows[:1]
    for row in rows:
        poster.db.update_post(row["id"], status=dbm.POST_SCHEDULED, attempts=0, scheduled_at=time.time(),
                              next_try_at=None, error=None)
        fresh = next(p for p in poster.db.posts_for_clip(row["clip_id"]) if p["id"] == row["id"])
        poster.post(fresh)
        after = next(p for p in poster.db.posts_for_clip(row["clip_id"]) if p["id"] == row["id"])
        print(f"clip {row['clip_id']} -> {row['platform']}: {after['status']}"
              + (f" {after['url']}" if after["url"] else "") + (f"\n  {after['error']}" if after["error"] else ""))


def cmd_make_cover(args) -> None:
    """Turn any image into the fixed cover: 1080x1920 (9:16) with a blurred fill behind it, JPEG
    under 2 MB (YouTube's thumbnail limit)."""
    from pathlib import Path

    from .media import run

    cfg = load_config()
    src = Path(args.image).expanduser()
    if not src.exists():
        sys.exit(f"Not found: {src}")
    out = cfg.cover_image
    out.parent.mkdir(parents=True, exist_ok=True)
    graph = ("[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=30:5[bg];"
             "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
             "[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuvj420p")
    for q in (2, 4, 6, 9, 13):
        run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-filter_complex", graph, "-frames:v", "1",
             "-q:v", str(q), str(out)])
        if out.stat().st_size < 1_900_000:
            break
    print(f"Cover saved: {out} ({out.stat().st_size // 1024} KB, 1080x1920)")
    print("YouTube uses it from the next upload. For Instagram, put this file at a public link and set "
          "COVER_IMAGE_URL in .env.")


def cmd_tiktok_login(_args) -> None:
    import httpx

    from .publish import PublishError
    from .publish import tiktok as tt

    cfg = load_config()
    if not (cfg.tiktok_client_key and cfg.tiktok_client_secret):
        sys.exit("Put TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET (from developers.tiktok.com) in .env first.")
    state = tt.new_state()
    print("1. Open this link in your browser and log in with the TikTok account to post to:\n")
    print(tt.auth_url(cfg, state))
    print(f"\n2. After you allow access you land on {cfg.tiktok_redirect_uri}.")
    print("   Tap Copy there (or copy the whole address bar) and paste it here.\n")
    for _ in range(3):
        try:
            code = tt.code_from_paste(input("Address: "), state)
            break
        except PublishError as exc:
            print(f"  {exc}")
    else:
        sys.exit("Run the command again when you have the address.")
    try:
        with httpx.Client(timeout=60) as http:
            tt.save_token(cfg.tiktok_token_path, tt.exchange_code(http, cfg, code))
            info = tt.TikTok(cfg, http=http).creator()
    except PublishError as exc:
        sys.exit(f"Login failed: {exc}")
    mode = ("as drafts in the TikTok app inbox" if cfg.tiktok_mode != "direct"
            else f"directly, as {cfg.tiktok_privacy}")
    print(f"\nLogged in as @{info.get('creator_username', '?')}. Approved clips go to TikTok {mode}.")
    print("Allowed privacy for this account: " + ", ".join(info.get("privacy_level_options") or ["?"]))


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

    r = sub.add_parser("reset-posts", help="Approved clips back to review, booked posts cancelled")
    r.add_argument("-y", "--yes", action="store_true", help="Don't ask for confirmation")
    r.set_defaults(func=cmd_reset_posts)

    f = sub.add_parser("retry-failed", help="Post failed posts again right now")
    f.add_argument("--platform", choices=["instagram", "youtube", "tiktok"])
    f.add_argument("--first", action="store_true", help="Only the oldest one (a quick test)")
    f.set_defaults(func=cmd_retry_failed)

    m = sub.add_parser("make-cover", help="Make the cover image used for every clip")
    m.add_argument("image")
    m.set_defaults(func=cmd_make_cover)

    tk = sub.add_parser("tiktok-login", help="Let ClipFarm post to your TikTok account (once)")
    tk.set_defaults(func=cmd_tiktok_login)

    c = sub.add_parser("cleanup", help="Delete clips older than the retention window")
    c.set_defaults(func=cmd_cleanup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
