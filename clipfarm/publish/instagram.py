"""Instagram Reels via the Instagram API with Instagram Login (no Facebook Page needed).

Flow: create a REELS container pointing at a public video URL (Instagram downloads it), wait
until the container is FINISHED, publish it, then read back the permalink.

Access token: a long-lived token (60 days) generated in the Meta App Dashboard goes in .env as
INSTAGRAM_ACCESS_TOKEN. It is refreshed automatically (refresh_access_token) once it is older
than REFRESH_AFTER and kept in the database; regenerating it in the dashboard and putting the new
one in .env takes over again.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)

REFRESH_AFTER = 20 * 86400       # refresh tokens older than 20 days (they live 60, need >= 24 h)
POLL_INTERVAL = 10
POLL_TIMEOUT = 900               # Instagram usually processes a 1-minute Reel in under a minute
CAPTION_LIMIT = 2200
HASHTAG_LIMIT = 30

# Graph API error codes worth retrying later (throttling / temporary trouble).
TRANSIENT_CODES = {1, 2, 4, 17, 32, 341, 613, 9004, 2207001, 2207003}


class InstagramError(RuntimeError):
    def __init__(self, message: str, transient: bool, code: int | None = None):
        super().__init__(message)
        self.transient = transient
        self.code = code


def build_caption(title: str, hashtags: str, suffix: str = "") -> str:
    tags = [f"#{t.lstrip('#')}" for t in hashtags.split() if t.strip("#")][:HASHTAG_LIMIT]
    parts = [title.strip()]
    if tags:
        parts.append(" ".join(tags))
    if suffix.strip():
        parts.append(suffix.strip())
    return "\n\n".join(parts)[:CAPTION_LIMIT]


class Instagram:
    platform = "instagram"

    def __init__(self, cfg, db, http: httpx.Client | None = None, sleep=time.sleep, clock=time.time):
        self.cfg = cfg
        self.db = db
        self.base = cfg.instagram_api_base.rstrip("/")
        self.http = http or httpx.Client(timeout=httpx.Timeout(60.0))
        self._sleep = sleep
        self._clock = clock
        self._user_id: str | None = None

    # -- token ------------------------------------------------------------------------------

    def _env_fingerprint(self) -> str:
        return hashlib.sha256(self.cfg.instagram_access_token.encode()).hexdigest()[:16]

    @property
    def token(self) -> str:
        """The current token: the refreshed one from the DB, unless .env holds a new one."""
        if self.db.kv_get("instagram_token_source") != self._env_fingerprint():
            self.db.kv_set("instagram_token", self.cfg.instagram_access_token)
            self.db.kv_set("instagram_token_at", str(self._clock()))
            self.db.kv_set("instagram_token_source", self._env_fingerprint())
        return self.db.kv_get("instagram_token") or self.cfg.instagram_access_token

    def maybe_refresh(self) -> bool:
        """Refresh the long-lived token when it is older than REFRESH_AFTER. True if refreshed."""
        token = self.token
        age = self._clock() - float(self.db.kv_get("instagram_token_at") or 0)
        if age < REFRESH_AFTER:
            return False
        data = self._request("GET", f"{self.base}/refresh_access_token",
                             {"grant_type": "ig_refresh_token", "access_token": token}, auth=False)
        self.db.kv_set("instagram_token", data["access_token"])
        self.db.kv_set("instagram_token_at", str(self._clock()))
        log.info("Instagram token refreshed (valid %s more days)", int(data.get("expires_in", 0)) // 86400)
        return True

    # -- HTTP -------------------------------------------------------------------------------

    def _request(self, method: str, url: str, params: dict[str, Any], auth: bool = True) -> dict:
        if auth:
            params = {**params, "access_token": self.token}
        try:
            resp = self.http.request(method, url, params=params if method == "GET" else None,
                                     data=params if method == "POST" else None)
        except httpx.HTTPError as exc:
            raise InstagramError(f"network: {exc.__class__.__name__}", transient=True) from exc
        try:
            body = resp.json()
        except ValueError as exc:
            raise InstagramError(f"HTTP {resp.status_code}, not JSON", transient=resp.status_code >= 500) from exc
        err = body.get("error") if isinstance(body, dict) else None
        if err:
            code = err.get("code")
            msg = err.get("error_user_msg") or err.get("message") or "unknown error"
            transient = bool(err.get("is_transient")) or code in TRANSIENT_CODES or resp.status_code >= 500
            if code == 190:
                msg = f"access token invalid or expired ({msg}). Generate a new one in the Meta dashboard and put it in .env"
            raise InstagramError(f"Instagram: {msg} (code {code})", transient=transient, code=code)
        if resp.status_code >= 400:
            raise InstagramError(f"Instagram HTTP {resp.status_code}", transient=resp.status_code >= 500)
        return body

    # -- API --------------------------------------------------------------------------------

    def me(self) -> dict:
        return self._request("GET", f"{self.base}/me", {"fields": "user_id,username"})

    def user_id(self) -> str:
        if self._user_id is None:
            self._user_id = str(self.me()["user_id"])
        return self._user_id

    def publishing_limit(self) -> dict:
        data = self._request("GET", f"{self.base}/{self.user_id()}/content_publishing_limit",
                             {"fields": "quota_usage,config"})
        rows = data.get("data") or [{}]
        return rows[0]

    def publish_reel(self, video_url: str, caption: str) -> tuple[str, str]:
        """Returns (media id, permalink)."""
        uid = self.user_id()
        container = self._request("POST", f"{self.base}/{uid}/media", {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": "true",
        })["id"]
        deadline = self._clock() + POLL_TIMEOUT
        while True:
            status = self._request("GET", f"{self.base}/{container}", {"fields": "status_code,status"})
            code = status.get("status_code")
            if code == "FINISHED":
                break
            if code in ("ERROR", "EXPIRED"):
                raise InstagramError(f"Instagram could not process the video: {status.get('status') or code}",
                                     transient=False)
            if self._clock() > deadline:
                raise InstagramError("Instagram took too long to process the video", transient=True)
            self._sleep(POLL_INTERVAL)
        media_id = self._request("POST", f"{self.base}/{uid}/media_publish", {"creation_id": container})["id"]
        try:
            link = self._request("GET", f"{self.base}/{media_id}", {"fields": "permalink"}).get("permalink", "")
        except InstagramError:
            link = ""  # posted fine; the link is a nicety
        return str(media_id), link
