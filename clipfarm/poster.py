"""Poster process: publishes approved clips at their scheduled slots.

    python -m clipfarm.poster

A separate process from the worker, so posts go out on time even while a long video is being
processed. Reports every post (or failure) in Telegram as a reply to the clip.
"""

from __future__ import annotations

import logging
import signal
import time
from pathlib import Path

from . import db as dbm
from .config import load_config
from .logsetup import setup_logging
from .notify import Telegram, TelegramError
from .publish.instagram import Instagram, InstagramError, build_caption
from .storage import R2, clip_key

log = logging.getLogger("clipfarm.poster")

MAX_ATTEMPTS = 4
RETRY_BASE = 300          # 5 min, doubling: 5, 10, 20 min
TICK = 30
MAINTENANCE_EVERY = 6 * 3600
NAMES = {"instagram": "Instagram"}


class Poster:
    def __init__(self, cfg=None, db=None, tg=None, publishers=None, clock=time.time, r2=None):
        self.cfg = cfg or load_config()
        self.db = db or dbm.Database(self.cfg.db_path)
        self.tg = tg if tg is not None else (
            Telegram(self.cfg.telegram_token, self.cfg.telegram_api_base) if self.cfg.telegram_enabled else None)
        self.publishers = publishers if publishers is not None else (
            {"instagram": Instagram(self.cfg, self.db)} if self.cfg.instagram_access_token else {})
        self._r2 = r2
        self._clock = clock
        self.running = True
        self.last_maintenance = 0.0

    # -- helpers --------------------------------------------------------------------------------

    @property
    def r2(self):
        if self._r2 is None and self.cfg.r2_enabled:
            self._r2 = R2(self.cfg)
        return self._r2

    def public_url(self, clip) -> str:
        """Instagram downloads the video itself, so the clip must be at a public URL (R2)."""
        if clip["public_url"]:
            return clip["public_url"]
        if self.r2 is None or not self.cfg.r2_public_base_url:
            raise InstagramError("Instagram needs the clip at a public URL: set up Cloudflare R2 "
                                 "(R2_* settings incl. R2_PUBLIC_BASE_URL) in .env", transient=False)
        local = Path(clip["local_path"] or "")
        if not local.exists():
            raise InstagramError(f"clip file is gone ({local.name}); it may be older than the retention period",
                                 transient=False)
        key = clip["r2_key"] or clip_key(clip["job_id"], clip["idx"], created=clip["created_at"])
        url = self.r2.upload(local, key, "video/mp4")
        self.db.update_clip(clip["id"], r2_key=key, public_url=url)
        return url

    def notify(self, clip, text: str) -> None:
        if not self.tg:
            return
        job = self.db.get_job(clip["job_id"])
        chat = int((job["chat_id"] if job else None) or self.cfg.telegram_owner_id)
        try:
            self.tg.send_message(chat, text, reply_to=clip["telegram_message_id"])
        except (TelegramError, OSError) as exc:
            log.warning("Could not report to Telegram: %s", exc)

    # -- posting ----------------------------------------------------------------------------------

    def post(self, row) -> None:
        if not self.db.claim_post(row["id"]):
            return  # cancelled or taken meanwhile
        clip = self.db.get_clip(row["clip_id"])
        name = NAMES.get(row["platform"], row["platform"])
        publisher = self.publishers.get(row["platform"])
        try:
            if clip is None or clip["status"] not in (dbm.APPROVED, dbm.POSTED):
                self.db.update_post(row["id"], status=dbm.POST_CANCELLED, error="clip no longer approved")
                return
            if publisher is None:
                raise InstagramError(f"{name} isn't connected (no credentials in .env)", transient=False)
            caption = build_caption(clip["title"], clip["hashtags"], self.cfg.caption_suffix)
            media_id, link = publisher.publish_reel(self.public_url(clip), caption)
        except InstagramError as exc:
            self._failed(row, clip, name, exc)
            return
        except Exception as exc:  # unexpected: treat as temporary, keep the poster alive
            log.exception("Posting %s to %s failed", row["clip_id"], name)
            self._failed(row, clip, name, InstagramError(f"{exc.__class__.__name__}: {exc}", transient=True))
            return
        self.db.update_post(row["id"], status=dbm.POST_POSTED, external_id=media_id, url=link,
                            posted_at=self._clock(), error=None)
        log.info("Posted clip %s to %s: %s", clip["id"], name, link or media_id)
        if all(p["status"] in (dbm.POST_POSTED, dbm.POST_CANCELLED) for p in self.db.posts_for_clip(clip["id"])):
            self.db.update_clip(clip["id"], status=dbm.POSTED)
        self.notify(clip, f"📤 Posted to {name}: {link or 'done'}\n{clip['title']}")

    def _failed(self, row, clip, name: str, exc: InstagramError) -> None:
        attempts = row["attempts"] + 1
        if exc.transient and attempts < MAX_ATTEMPTS:
            wait = RETRY_BASE * 2 ** (attempts - 1)
            self.db.update_post(row["id"], status=dbm.POST_SCHEDULED, attempts=attempts,
                                next_try_at=self._clock() + wait, error=str(exc)[:500])
            log.warning("%s post of clip %s failed (%s); retrying in %d min", name, row["clip_id"], exc, wait // 60)
            return
        self.db.update_post(row["id"], status=dbm.POST_FAILED, attempts=attempts, error=str(exc)[:500])
        log.warning("%s post of clip %s failed for good: %s", name, row["clip_id"], exc)
        if clip is not None:
            self.notify(clip, f"⚠️ Couldn't post to {name}: {exc}\n"
                              "Tap ↩️ Undo then ✅ Post on the clip to try again.")

    # -- loop ---------------------------------------------------------------------------------------

    def maintenance(self) -> None:
        self.last_maintenance = self._clock()
        for publisher in self.publishers.values():
            try:
                publisher.maybe_refresh()
            except InstagramError as exc:
                log.warning("Token refresh failed: %s", exc)

    def tick(self) -> int:
        if self._clock() - self.last_maintenance > MAINTENANCE_EVERY:
            self.maintenance()
        due = self.db.due_posts(self._clock())
        for row in due:
            self.post(row)
        return len(due)

    def run(self) -> None:
        n = self.db.requeue_stuck_posts()
        if n:
            log.info("Re-queued %d posts interrupted by a restart", n)
        log.info("Poster ready (platforms: %s, %d/day, %s %s)", ", ".join(self.publishers) or "none",
                 self.cfg.posts_per_day, self.cfg.post_window, self.cfg.post_tz)
        while self.running:
            self.tick()
            for _ in range(TICK):
                if not self.running:
                    break
                time.sleep(1)

    def stop(self, *_):
        self.running = False


def main() -> None:
    setup_logging()
    poster = Poster()
    signal.signal(signal.SIGTERM, poster.stop)
    signal.signal(signal.SIGINT, poster.stop)
    poster.run()


if __name__ == "__main__":
    main()
