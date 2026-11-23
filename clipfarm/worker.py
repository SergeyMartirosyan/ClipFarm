"""Worker process: takes queued links one at a time and runs the pipeline.

    python -m clipfarm.worker
"""

from __future__ import annotations

import logging
import signal
import time

from . import db as dbm
from . import messages
from .config import load_config
from .notify import Telegram, TelegramError
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

    # -- telegram helpers (never let a Telegram hiccup fail a job) ----------------

    def _chat(self, job) -> int:
        return int(job["chat_id"] or self.cfg.telegram_owner_id)

    def _show_status(self, job_id: int, detail: str = "", extra: str = "") -> None:
        if not self.tg:
            return
        job = self.db.get_job(job_id)
        text = messages.job_text(job, detail, extra)
        try:
            if job["status_message_id"]:
                self.tg.edit_message(self._chat(job), job["status_message_id"], text)
            else:
                mid = self.tg.send_message(self._chat(job), text)
                self.db.update_job(job_id, status_message_id=mid)
        except (TelegramError, OSError) as exc:
            log.warning("Status update failed: %s", exc)

    def _send_clip(self, job, clip_id: int, rendered: RenderedClip) -> None:
        if not self.tg:
            return
        row = self.db.get_clip(clip_id)
        caption = messages.clip_caption(row)
        keyboard = messages.buttons_as_json(messages.clip_buttons(row))
        try:
            mid = self.tg.send_video(self._chat(job), rendered.preview_path, caption, keyboard)
        except (TelegramError, OSError) as exc:
            log.warning("Video upload to Telegram failed (%s); sending text instead", exc)
            link = rendered.public_url or str(rendered.final_path)
            mid = self.tg.send_message(self._chat(job), f"{caption}\n\n▶️ {link}", keyboard)
        self.db.update_clip(clip_id, telegram_message_id=mid)

    # -- job ------------------------------------------------------------------------

    def handle(self, job) -> None:
        job_id = job["id"]
        log.info("Job #%d: %s", job_id, job["url"])
        sent = 0

        def on_status(status: str, detail: str) -> None:
            self.db.update_job(job_id, status=status)
            self._show_status(job_id, detail)

        def on_source(src) -> None:
            self.db.update_job(job_id, source_title=src.title, uploader=src.uploader, duration=src.duration)

        def on_clip(rc: RenderedClip) -> None:
            nonlocal sent
            c = rc.candidate
            clip_id = self.db.add_clip(
                job_id=job_id, idx=rc.idx, total=rc.total, start=c.start, end=c.end,
                title=c.title, hashtags=" ".join(c.hashtags), reason=c.reason,
                llm_score=c.llm_score, signal_score=c.signal_score, final_score=c.final_score,
                judge_model=c.model, layout=rc.layout, local_path=str(rc.final_path),
                r2_key=rc.r2_key, public_url=rc.public_url,
            )
            self._send_clip(job, clip_id, rc)
            sent += 1

        try:
            process(
                job["url"], job_id, self.cfg,
                transcriber=self.transcriber, db=self.db,
                on_status=on_status, on_clip=on_clip, on_source=on_source,
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
        removed = cleanup_local(self.cfg.clips_dir, self.cfg.local_retention_days)
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
            job = self.db.claim_next_job()
            if job is None:
                time.sleep(5)
                continue
            self.handle(job)

    def stop(self, *_):
        log.info("Stopping after the current step…")
        self.running = False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    worker = Worker()
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    worker.run()


if __name__ == "__main__":
    main()
