"""Minimal synchronous Telegram Bot API client for the worker process, with a circuit breaker.

When Telegram refuses or hangs (rate limit, failed bot login, network trouble), hammering it
makes things worse: every request after a login refusal can start a new login attempt. So after
a failure the client "opens the circuit" for a while and fails fast with TelegramUnavailable;
the worker keeps processing and re-delivers what it couldn't send once the circuit closes.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 49 * 1024 * 1024  # bots can upload up to 50 MB
FLOOR_START = 60.0
FLOOR_MAX = 1800.0
SHORT_WAIT = 5  # retry_after up to this is normal per-chat pacing: sleep it off once


class TelegramError(RuntimeError):
    pass


class TelegramUnavailable(TelegramError):
    """Telegram is rate-limiting, refusing or not answering; try again after `retry_at`."""

    def __init__(self, message: str, retry_at: float):
        super().__init__(message)
        self.retry_at = retry_at


class Telegram:
    def __init__(self, token: str, api_base: str = "https://api.telegram.org", clock=time.time, sleep=time.sleep):
        self.base = f"{api_base.rstrip('/')}/bot{token}"
        self.http = httpx.Client(timeout=httpx.Timeout(90.0, write=300.0))
        self.blocked_until = 0.0
        self.floor = FLOOR_START
        self._clock = clock
        self._sleep = sleep

    @property
    def available(self) -> bool:
        return self._clock() >= self.blocked_until

    def _trip(self, retry_after: float, why: str) -> TelegramUnavailable:
        wait = max(retry_after, self.floor) * random.uniform(1.0, 1.1)
        self.blocked_until = self._clock() + wait
        self.floor = min(self.floor * 2, FLOOR_MAX)
        log.warning("Telegram unavailable (%s); pausing Telegram calls for %.0fs", why, wait)
        return TelegramUnavailable(f"Telegram unavailable: {why}", self.blocked_until)

    def _call(self, method: str, data: dict, video: Path | None = None) -> dict:
        if not self.available:
            raise TelegramUnavailable("Telegram paused after an earlier failure", self.blocked_until)
        url = f"{self.base}/{method}"
        for attempt in range(2):
            try:
                if video is not None:
                    with open(video, "rb") as fh:  # reopened per attempt: an upload consumes the handle
                        resp = self.http.post(url, data=data, files={"video": (Path(video).name, fh, "video/mp4")})
                else:
                    resp = self.http.post(url, data=data)
            except httpx.HTTPError as exc:
                raise self._trip(0, f"{method}: {exc.__class__.__name__}") from exc
            try:
                body = resp.json()
            except ValueError as exc:  # a proxy/relay error page
                raise self._trip(0, f"{method}: HTTP {resp.status_code}, not a Telegram reply") from exc
            if body.get("ok"):
                self.floor = FLOOR_START
                return body["result"]
            if body.get("error_code") == 429:
                retry = int((body.get("parameters") or {}).get("retry_after") or 0)
                if attempt == 0 and 0 < retry <= SHORT_WAIT:
                    self._sleep(retry + 1)
                    continue
                raise self._trip(retry, f"{method}: {body.get('description')}")
            raise TelegramError(f"{method}: {body.get('description')}")
        raise self._trip(SHORT_WAIT, f"{method}: still rate-limited after a short wait")

    def send_message(self, chat_id: int, text: str, keyboard: dict | None = None) -> int:
        data = {"chat_id": chat_id, "text": text, "disable_web_page_preview": "true"}
        if keyboard:
            data["reply_markup"] = json.dumps(keyboard)
        return int(self._call("sendMessage", data)["message_id"])

    def edit_message(self, chat_id: int, message_id: int, text: str) -> None:
        try:
            self._call(
                "editMessageText",
                {"chat_id": chat_id, "message_id": message_id, "text": text, "disable_web_page_preview": "true"},
            )
        except TelegramUnavailable:
            raise
        except TelegramError as exc:
            if "message is not modified" not in str(exc):
                raise

    def send_video(self, chat_id: int, path: Path, caption: str, keyboard: dict | None = None) -> int:
        if Path(path).stat().st_size > MAX_UPLOAD_BYTES:
            raise TelegramError(f"{Path(path).name} is over Telegram's 50 MB bot limit")
        data = {"chat_id": chat_id, "caption": caption, "supports_streaming": "true"}
        if keyboard:
            data["reply_markup"] = json.dumps(keyboard)
        return int(self._call("sendVideo", data, video=Path(path))["message_id"])
