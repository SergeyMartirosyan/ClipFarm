"""Minimal synchronous Telegram Bot API client for the worker process."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 49 * 1024 * 1024  # bots can upload up to 50 MB


class TelegramError(RuntimeError):
    pass


class Telegram:
    def __init__(self, token: str, api_base: str = "https://api.telegram.org"):
        self.base = f"{api_base.rstrip('/')}/bot{token}"
        self.http = httpx.Client(timeout=httpx.Timeout(60.0, write=300.0))

    def _call(self, method: str, data: dict, files: dict | None = None) -> dict:
        for attempt in range(3):
            resp = self.http.post(f"{self.base}/{method}", data=data, files=files)
            body = resp.json()
            retry = (body.get("parameters") or {}).get("retry_after")
            if resp.status_code == 429 and retry and attempt < 2 and not files:
                log.warning("Telegram rate limit on %s; waiting %ss", method, retry)
                time.sleep(min(int(retry), 120) + 1)
                continue
            break
        if not body.get("ok"):
            raise TelegramError(f"{method}: {body.get('description')}")
        return body["result"]

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
        except TelegramError as exc:
            if "message is not modified" not in str(exc):
                raise

    def send_video(self, chat_id: int, path: Path, caption: str, keyboard: dict | None = None) -> int:
        if Path(path).stat().st_size > MAX_UPLOAD_BYTES:
            raise TelegramError(f"{path.name} is over Telegram's 50 MB bot limit")
        data = {"chat_id": chat_id, "caption": caption, "supports_streaming": "true"}
        if keyboard:
            data["reply_markup"] = json.dumps(keyboard)
        with open(path, "rb") as fh:
            result = self._call("sendVideo", data, files={"video": (Path(path).name, fh, "video/mp4")})
        return int(result["message_id"])
