"""Logging for the bot, worker and CLI, with secrets kept out of the logs."""

from __future__ import annotations

import logging
import re

# Telegram bot tokens appear in API URLs as /bot<id>:<secret>/.
_TOKEN_RE = re.compile(r"(bot\d+):[A-Za-z0-9_-]{20,}")
_SECRET_PARAM_RE = re.compile(r"((?:client_secret|key|api_key|access_token)=)[^&\s]+", re.IGNORECASE)


def redact(text: str) -> str:
    return _SECRET_PARAM_RE.sub(r"\1***", _TOKEN_RE.sub(r"\1:***", text))


class RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        clean = redact(msg)
        if clean != msg:
            record.msg, record.args = clean, None
        return True


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # httpx logs every request URL at INFO, and Telegram URLs contain the bot token.
    for noisy in ("httpx", "httpcore", "telegram.ext", "google_genai", "urllib3", "botocore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Belt and braces: scrub anything token-like that still reaches a handler.
    for handler in logging.getLogger().handlers:
        handler.addFilter(RedactFilter())
