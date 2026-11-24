"""Command line, for testing without Telegram.

    python -m clipfarm.cli process <url> [--clips N] [--keep-work]
    python -m clipfarm.cli cleanup
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

    c = sub.add_parser("cleanup", help="Delete clips older than the retention window")
    c.set_defaults(func=cmd_cleanup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
