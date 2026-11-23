"""Minimal synchronous Telegram Bot API client for the worker process."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 49 * 1024 * 1024  # bots can upload up to 50 MB


class TelegramError(RuntimeError):
    pass


class Telegram:
    def __init__(self, token: str):
        self.base = f"https://api.telegram.org/bot{token}"
        self.http = httpx.Client(timeout=httpx.Timeout(30.0, write=300.0))

    def _call(self, method: str, data: dict, files: dict | None = None) -> dict:
        resp = self.http.post(f"{self.base}/{method}", data=data, files=files)
        body = resp.json()
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
