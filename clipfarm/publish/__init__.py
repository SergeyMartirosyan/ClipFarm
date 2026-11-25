"""Posting approved clips to social platforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


class PublishError(RuntimeError):
    """transient: worth retrying later. retry_after: seconds to wait instead of the usual backoff."""

    def __init__(self, message: str, transient: bool, code=None, retry_after: float | None = None):
        super().__init__(message)
        self.transient = transient
        self.code = code
        self.retry_after = retry_after


@dataclass
class PostItem:
    title: str
    text: str                       # post text: caption (or title) + hashtags + suffix
    hashtags: list[str]
    local_path: Path
    cover_offset: float | None = None
    # Uploads the clip to R2 and returns its public URL; None when R2 isn't set up.
    public_url: Callable[[], str] | None = None


@dataclass
class Published:
    id: str
    url: str = ""
    note: str = ""                  # anything the group should know, e.g. "went up as Private"
    extra: dict = field(default_factory=dict)
