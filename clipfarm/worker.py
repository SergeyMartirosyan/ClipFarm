"""Worker process: takes queued links one at a time and runs the pipeline.

    python -m clipfarm.worker
"""

from __future__ import annotations

import logging
import signal
import time
from pathlib import Path

from . import db as dbm
from . import messages
from .config import load_config
from .logsetup import setup_logging
from .notify import Telegram, TelegramError, TelegramUnavailable
from .pipeline.judge import JudgeBudgetExceeded
from .pipeline.run import RenderedClip, process
from .pipeline.transcribe import Transcriber
from .storage import R2, cleanup_local

log = logging.getLogger("clipfarm.worker")

CLEANUP_EVERY = 6 * 3600


class Worker:
    def __init__(self):
        self.cfg = load_config()
        self.db = dbm.Database(self.cfg.db_path)
        self.tg = Telegram(self.cfg.telegram_token, self.cfg.telegram_api_base) if self.cfg.telegram_enabled else None
        self.transcriber = Transcriber(
            self.cfg.whisper_model, self.cfg.whisper_compute_type, self.cfg.whisper_threads
        )
        self.running = True
        self.last_cleanup = 0.0
        self.stale_status: dict[int, tuple[str, str]] = {}

    # -- telegram helpers (never let a Telegram hiccup fail a job) ----------------

    def _chat(self, job) -> int:
        return int(job["chat_id"] or self.cfg.telegram_chat_id or self.cfg.telegram_owner_id)

    def _show_status(self, job_id: int, detail: str = "", extra: str = "") -> None:
        """Edit the job's status message. While Telegram is unavailable, remember the latest
        status instead; it is pushed once Telegram answers again (flush_telegram)."""
        if not self.tg:
            return
        if not self.tg.available:
            self.stale_status[job_id] = (detail, extra)
            return
        job = self.db.get_job(job_id)
        text = messages.job_text(job, detail, extra)
        try:
            if job["status_message_id"]:
                self.tg.edit_message(self._chat(job), job["status_message_id"], text)
            else:
                mid = self.tg.send_message(self._chat(job), text)
                self.db.update_job(job_id, status_message_id=mid)
            self.stale_status.pop(job_id, None)
        except TelegramUnavailable:
            self.stale_status[job_id] = (detail, extra)
        except (TelegramError, OSError) as exc:
            log.warning("Status update failed: %s", exc)

    def _deliver(self, clip) -> bool:
        """Send one clip for review. False = Telegram is unavailable, stop delivering for now."""
        caption = messages.clip_caption(clip)
        keyboard = messages.buttons_as_json(messages.clip_buttons(clip))
        chat = int(clip["chat_id"] or self.cfg.telegram_chat_id or self.cfg.telegram_owner_id)
        final = Path(clip["local_path"] or "")
        preview = final.with_name(final.stem + ".preview.mp4")
        try:
            mid = self.tg.send_video(chat, preview, caption, keyboard)
        except TelegramUnavailable:
            return False
        except (TelegramError, OSError) as exc:
            log.warning("Video upload for clip %s failed (%s); sending text instead", clip["id"], exc)
            link = clip["public_url"] or str(final)
            try:
                mid = self.tg.send_message(chat, f"{caption}\n\n▶️ {link}", keyboard)
            except TelegramUnavailable:
                return False
            except (TelegramError, OSError) as exc2:
                log.warning("Could not deliver clip %s: %s", clip["id"], exc2)
                self.db.update_clip(clip["id"], deliver_attempts=clip["deliver_attempts"] + 1)
                return True
        self.db.update_clip(clip["id"], telegram_message_id=mid)
        return True

    def flush_telegram(self) -> None:
        """Push stale status messages and undelivered clips, while Telegram is reachable."""
        if not self.tg or not self.tg.available:
            return
        for job_id, (detail, extra) in list(self.stale_status.items()):
            self._show_status(job_id, detail, extra)
            if not self.tg.available:
                return
        for clip in self.db.undelivered_clips():
            if not self._deliver(clip):
                return

    # -- job ------------------------------------------------------------------------

    def handle(self, job) -> None:
        job_id = job["id"]
        log.info("Job #%d: %s", job_id, job["url"])
        sent = 0

        def on_status(status: str, detail: str) -> None:
            log.info("Job #%d [%s] %s", job_id, status, detail)
            self.db.update_job(job_id, status=status)
            self._show_status(job_id, detail)
            self.flush_telegram()

        def on_source(src) -> None:
            self.db.update_job(job_id, source_title=src.title, uploader=src.uploader, duration=src.duration)

        def on_clip(rc: RenderedClip) -> None:
            nonlocal sent
            c = rc.candidate
            clip_id = self.db.add_clip(
                job_id=job_id, idx=rc.idx, total=rc.total, start=c.start, end=c.end,
                title=c.title, hashtags=" ".join(c.hashtags), reason=c.reason, vision_note=c.vision_note,
                llm_score=c.llm_score, signal_score=c.signal_score, final_score=c.final_score,
                judge_model=c.model, layout=rc.layout, local_path=str(rc.final_path),
                r2_key=rc.r2_key, public_url=rc.public_url,
            )
            sent += 1
            self.flush_telegram()  # delivers this clip, plus any that failed earlier

        try:
            process(
                job["url"], job_id, self.cfg,
                transcriber=self.transcriber, db=self.db,
                on_status=on_status, on_clip=on_clip, on_source=on_source,
                attempt=int(job["attempts"] or 1),
            )
        except JudgeBudgetExceeded as exc:
            self._fail(job_id, f"{exc}. Gemini was unavailable; send the link again later.")
            return
        except Exception as exc:
            log.exception("Job #%d failed", job_id)
            self._fail(job_id, str(exc) or exc.__class__.__name__)
            return
        self.db.update_job(job_id, status=dbm.DONE)
        self._show_status(job_id, extra=f"\n{sent} clips sent for review below 👇")

    def _fail(self, job_id: int, error: str) -> None:
        self.db.update_job(job_id, status=dbm.FAILED, error=error[:2000])
        self._show_status(job_id)

    # -- housekeeping ----------------------------------------------------------------

    def cleanup(self) -> None:
        self.last_cleanup = time.time()
        removed = cleanup_local(self.cfg.clips_dir, self.cfg.local_retention_days, self.cfg.work_dir)
        log.info("Local cleanup: removed %d old job folders", removed)
        if self.cfg.r2_enabled:
            try:
                n = R2(self.cfg).cleanup(self.cfg.r2_retention_days)
                log.info("R2 cleanup: deleted %d old objects", n)
            except Exception as exc:
                log.warning("R2 cleanup failed: %s", exc)

    def run(self) -> None:
        requeued = self.db.requeue_stuck_jobs()
        if requeued:
            log.info("Re-queued %d jobs interrupted by a restart", requeued)
        log.info("Worker ready (telegram=%s, r2=%s)", bool(self.tg), self.cfg.r2_enabled)
        while self.running:
            if time.time() - self.last_cleanup > CLEANUP_EVERY:
                self.cleanup()
            self.flush_telegram()
            job = self.db.claim_next_job()
            if job is None:
                time.sleep(5)
                continue
            self.handle(job)

    def stop(self, *_):
        log.info("Stopping after the current step…")
        self.running = False


def main() -> None:
    setup_logging()
    worker = Worker()
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    worker.run()


if __name__ == "__main__":
    main()
