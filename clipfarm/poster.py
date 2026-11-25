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
from .publish import PostItem, PublishError
from .publish.instagram import Instagram, build_caption
from .storage import R2, clip_key

log = logging.getLogger("clipfarm.poster")

MAX_ATTEMPTS = 4
RETRY_BASE = 300          # 5 min, doubling: 5, 10, 20 min
TICK = 30
MAINTENANCE_EVERY = 6 * 3600
NAMES = {"instagram": "Instagram", "youtube": "YouTube"}
LOGIN_WARN_EVERY = 12 * 3600


class Poster:
    def __init__(self, cfg=None, db=None, tg=None, publishers=None, clock=time.time, r2=None):
        self.cfg = cfg or load_config()
        self.db = db or dbm.Database(self.cfg.db_path)
        self.tg = tg if tg is not None else (
            Telegram(self.cfg.telegram_token, self.cfg.telegram_api_base) if self.cfg.telegram_enabled else None)
        self._fixed_publishers = publishers is not None
        self.publishers = publishers if publishers is not None else {}
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

    def publisher(self, platform: str):
        """Built on first use, so a YouTube login made after start-up is picked up without a restart."""
        if platform not in self.publishers and not self._fixed_publishers:
            if platform == "instagram" and self.cfg.instagram_access_token:
                self.publishers[platform] = Instagram(self.cfg, self.db)
            elif platform == "youtube" and self.cfg.youtube_token_path.exists():
                from .publish.youtube import YouTube

                self.publishers[platform] = YouTube(self.cfg, self.db)
        return self.publishers.get(platform)

    def public_url(self, clip) -> str:
        """The clip at a public R2 URL (uploaded on demand)."""
        if clip["public_url"]:
            return clip["public_url"]
        key = clip["r2_key"] or clip_key(clip["job_id"], clip["idx"], created=clip["created_at"])
        url = self.r2.upload(Path(clip["local_path"]), key, "video/mp4")
        self.db.update_clip(clip["id"], r2_key=key, public_url=url)
        return url

    def item_for(self, clip, platform: str) -> PostItem:
        local = Path(clip["local_path"] or "")
        has_url = bool(clip["public_url"]) or (self.r2 is not None and bool(self.cfg.r2_public_base_url))
        if not local.exists() and not (platform == "instagram" and clip["public_url"]):
            raise PublishError(f"clip file is gone ({local.name}); it may be older than the retention period",
                               transient=False)
        return PostItem(
            title=clip["title"],
            text=build_caption(clip["caption"] or clip["title"], clip["hashtags"], self.cfg.caption_suffix),
            hashtags=clip["hashtags"].split(),
            local_path=local,
            cover_offset=clip["cover_offset"],
            public_url=(lambda: self.public_url(clip)) if has_url else None,
        )

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
        publisher = self.publisher(row["platform"])
        try:
            if clip is None or clip["status"] not in (dbm.APPROVED, dbm.POSTED):
                self.db.update_post(row["id"], status=dbm.POST_CANCELLED, error="clip no longer approved")
                return
            if publisher is None:
                raise PublishError(f"{name} isn't connected", transient=False)
            done = publisher.publish(self.item_for(clip, row["platform"]))
        except PublishError as exc:
            self._failed(row, clip, name, exc)
            return
        except Exception as exc:  # unexpected: treat as temporary, keep the poster alive
            log.exception("Posting %s to %s failed", row["clip_id"], name)
            self._failed(row, clip, name, PublishError(f"{exc.__class__.__name__}: {exc}", transient=True))
            return
        link = done.url
        self.db.update_post(row["id"], status=dbm.POST_POSTED, external_id=done.id, url=link,
                            posted_at=self._clock(), error=done.note or None)
        log.info("Posted clip %s to %s: %s %s", clip["id"], name, link or done.id, done.note)
        if all(p["status"] in (dbm.POST_POSTED, dbm.POST_CANCELLED) for p in self.db.posts_for_clip(clip["id"])):
            self.db.update_clip(clip["id"], status=dbm.POSTED)
        self.notify(clip, f"📤 Posted to {name}: {link or 'done'}\n{clip['title']}"
                          + (f"\n{done.note}" if done.note else ""))

    def _failed(self, row, clip, name: str, exc: PublishError) -> None:
        attempts = row["attempts"] + 1
        if exc.transient and attempts < MAX_ATTEMPTS:
            wait = exc.retry_after or RETRY_BASE * 2 ** (attempts - 1)
            self.db.update_post(row["id"], status=dbm.POST_SCHEDULED, attempts=attempts,
                                next_try_at=self._clock() + wait, error=str(exc)[:500])
            log.warning("%s post of clip %s failed (%s); retrying in %d min", name, row["clip_id"], exc, int(wait // 60))
            return
        self.db.update_post(row["id"], status=dbm.POST_FAILED, attempts=attempts, error=str(exc)[:500])
        log.warning("%s post of clip %s failed for good: %s", name, row["clip_id"], exc)
        if clip is not None:
            self.notify(clip, f"⚠️ Couldn't post to {name}: {exc}\n"
                              "Tap ↩️ Undo then ✅ Post on the clip to try again.")

    # -- loop ---------------------------------------------------------------------------------------

    def maintenance(self) -> None:
        """Refresh tokens; tell the group (at most every 12 h) when a login needs redoing."""
        self.last_maintenance = self._clock()
        for platform in self.cfg.post_platforms if not self._fixed_publishers else list(self.publishers):
            publisher = self.publisher(platform)
            if publisher is None:
                continue
            try:
                publisher.maybe_refresh()
                self.db.kv_set(f"login_warned_{platform}", None)
            except PublishError as exc:
                log.warning("%s token check failed: %s", NAMES.get(platform, platform), exc)
                if not exc.transient:
                    self.warn_group(platform, f"⚠️ {NAMES.get(platform, platform)}: {exc}")

    def warn_group(self, platform: str, text: str) -> None:
        last = self.db.kv_get(f"login_warned_{platform}")
        if not self.tg or (last is not None and self._clock() - float(last) < LOGIN_WARN_EVERY):
            return
        chat = self.cfg.telegram_chat_id or self.cfg.telegram_owner_id
        try:
            self.tg.send_message(int(chat), text)
            self.db.kv_set(f"login_warned_{platform}", str(self._clock()))
        except (TelegramError, OSError) as exc:
            log.warning("Could not warn the group: %s", exc)

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
        log.info("Poster ready (platforms: %s, %d/day, %s %s)", ", ".join(self.cfg.post_platforms) or "none",
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
