"""Instagram Reels via the Instagram API with Instagram Login (no Facebook Page needed).

Flow: create a REELS container, give it the video, wait until the container is FINISHED,
publish it, then read back the permalink. The video goes either straight from this machine
(resumable upload to rupload.facebook.com, no extra service needed) or, when Cloudflare R2 is set
up, as a public URL Instagram downloads itself.

Two kinds of token work. IGAA... (Instagram Login) talks to graph.instagram.com. EAA... (what the
dashboard's "API setup with Unified login" / Facebook Login gives) talks to graph.facebook.com and
needs the Instagram account id in INSTAGRAM_USER_ID (shown under the account on that page); the
posting calls are the same. EAA tokens can't be refreshed without the app secret, so they are only
checked, and an expired one is reported in the group.

Access token: a long-lived token (60 days) generated in the Meta App Dashboard goes in .env as
INSTAGRAM_ACCESS_TOKEN. It is refreshed automatically (refresh_access_token) once it is older
than REFRESH_AFTER and kept in the database; regenerating it in the dashboard and putting the new
one in .env takes over again.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from . import PostItem, PublishError, Published

log = logging.getLogger(__name__)

REFRESH_AFTER = 20 * 86400       # refresh tokens older than 20 days (they live 60, need >= 24 h)
POLL_INTERVAL = 10
POLL_TIMEOUT = 900               # Instagram usually processes a 1-minute Reel in under a minute
CAPTION_LIMIT = 2200
HASHTAG_LIMIT = 30

# Graph API error codes worth retrying later (throttling / temporary trouble).
TRANSIENT_CODES = {1, 2, 4, 17, 32, 341, 613, 9004, 2207001, 2207003}


RUPLOAD = "https://rupload.facebook.com/ig-api-upload/{version}/{container}"
RUPLOAD_VERSION = "v21.0"        # only used if the container response carries no upload "uri"


class InstagramError(PublishError):
    pass


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
        self.http = http or httpx.Client(timeout=httpx.Timeout(60.0))
        self._sleep = sleep
        self._clock = clock
        self._user_id: str | None = None

    # -- token ------------------------------------------------------------------------------

    @property
    def facebook_token(self) -> bool:
        return self.token.startswith("EAA")

    @property
    def base(self) -> str:
        return (self.cfg.instagram_fb_api_base if self.facebook_token else self.cfg.instagram_api_base).rstrip("/")

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
        """Refresh the long-lived token when it is older than REFRESH_AFTER. True if refreshed.
        EAA tokens can't be refreshed this way: just check they still work (raises if not)."""
        token = self.token
        if self.facebook_token:
            self.me()
            return False
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
        if not self.facebook_token:
            return self._request("GET", f"{self.base}/me", {"fields": "user_id,username"})
        data = self._request("GET", f"{self.base}/{self.user_id()}", {"fields": "id,username"})
        return {"user_id": data.get("id"), "username": data.get("username")}

    def user_id(self) -> str:
        if self._user_id is None:
            if self.cfg.instagram_user_id:
                self._user_id = self.cfg.instagram_user_id.strip()
            elif self.facebook_token:
                self._user_id = self._find_account()
            else:
                self._user_id = str(self.me()["user_id"])
        return self._user_id

    def _find_account(self) -> str:
        """EAA token without INSTAGRAM_USER_ID: the Instagram account linked to the token's Pages."""
        try:
            pages = self._request("GET", f"{self.base}/me/accounts",
                                  {"fields": "instagram_business_account{id,username}"}).get("data") or []
        except InstagramError:
            pages = []
        for page in pages:
            acc = page.get("instagram_business_account") or {}
            if acc.get("id"):
                return str(acc["id"])
        raise InstagramError("This token starts with EAA: put the Instagram account id in .env as "
                             "INSTAGRAM_USER_ID (the number under the account in the Meta dashboard)",
                             transient=False)

    def publishing_limit(self) -> dict:
        data = self._request("GET", f"{self.base}/{self.user_id()}/content_publishing_limit",
                             {"fields": "quota_usage,config"})
        rows = data.get("data") or [{}]
        return rows[0]

    def publish(self, item: PostItem) -> Published:
        """Retries are duplicate-proof: the container of an attempt is remembered, and the next
        attempt first asks Instagram what became of it (already published, ready to publish, still
        processing) before uploading again. Meta sometimes answers an error and posts anyway."""
        slot = f"ig_container_{item.key}" if item.key else None
        remember = (lambda cid: self.db.kv_set(slot, cid)) if slot else (lambda cid: None)
        try:
            saved = self.db.kv_get(slot) if slot else None
            done = self._resume(saved) if saved else None
            if done is None:
                if slot:
                    self.db.kv_set(slot, None)
                if item.public_url is not None:
                    done = self.publish_reel(item.public_url(), item.text, item.cover_offset, remember)
                else:
                    done = self.publish_reel_file(Path(item.local_path), item.text, item.cover_offset, remember)
        except InstagramError as exc:
            if slot and not exc.transient:
                self.db.kv_set(slot, None)
            raise
        if slot:
            self.db.kv_set(slot, None)
        return Published(id=done[0], url=done[1])

    def _resume(self, container: str) -> tuple[str, str] | None:
        """What became of an earlier attempt's container. None: start over."""
        try:
            code = self._request("GET", f"{self.base}/{container}", {"fields": "status_code"}).get("status_code")
        except InstagramError as exc:
            if exc.transient:
                raise
            return None
        log.info("Instagram: earlier attempt's container %s is %s", container, code)
        if code == "PUBLISHED":
            return self._latest_media()
        if code in ("FINISHED", "IN_PROGRESS"):
            return self._finish(container)
        return None

    def _latest_media(self) -> tuple[str, str]:
        rows = self._request("GET", f"{self.base}/{self.user_id()}/media",
                             {"fields": "id,permalink", "limit": "1"}).get("data") or [{}]
        return str(rows[0].get("id", "")), rows[0].get("permalink", "")

    def _container_params(self, caption: str, cover_offset: float | None) -> dict:
        params = {"media_type": "REELS", "caption": caption, "share_to_feed": "true"}
        if self.cfg.cover_image_url:
            params["cover_url"] = self.cfg.cover_image_url  # the fixed cover wins over a video frame
        elif cover_offset is not None:
            params["thumb_offset"] = str(int(cover_offset * 1000))
        return params

    def publish_reel(self, video_url: str, caption: str, cover_offset: float | None = None,
                     on_container=None) -> tuple[str, str]:
        """Instagram fetches the video from a public URL. Returns (media id, permalink).
        cover_offset: seconds into the video for the cover frame."""
        params = self._container_params(caption, cover_offset) | {"video_url": video_url}
        container = self._request("POST", f"{self.base}/{self.user_id()}/media", params)["id"]
        if on_container:
            on_container(container)
        return self._finish(container)

    def publish_reel_file(self, path: Path, caption: str, cover_offset: float | None = None,
                          on_container=None) -> tuple[str, str]:
        """Sends the file from this machine (resumable upload): no public URL needed."""
        params = self._container_params(caption, cover_offset) | {"upload_type": "resumable"}
        body = self._request("POST", f"{self.base}/{self.user_id()}/media", params)
        container = body["id"]
        uri = body.get("uri") or RUPLOAD.format(version=RUPLOAD_VERSION, container=container)
        data = path.read_bytes()
        try:
            resp = self.http.post(uri, content=data, headers={
                "Authorization": f"OAuth {self.token}", "offset": "0", "file_size": str(len(data))},
                timeout=httpx.Timeout(600.0))
        except httpx.HTTPError as exc:
            raise InstagramError(f"network while uploading: {exc.__class__.__name__}", transient=True) from exc
        try:
            result = resp.json()
        except ValueError:
            result = {}
        if resp.status_code >= 400 or (isinstance(result, dict) and result.get("error")):
            # Keep Meta's whole answer: its short message ("Request processing failed") says little.
            raw = " ".join(resp.text.split())[:400] if resp.text else "(empty)"
            where = uri.split("?")[0].replace(container, "<container>")
            log.warning("Instagram upload to %s failed: HTTP %s %s", where, resp.status_code, raw)
            # "Request processing failed" (ProcessingFailedError) is a hiccup on Meta's side: the same
            # clip went through on a retry. Only other 4xx answers are final.
            hiccup = "processing failed" in raw.lower() or "ProcessingFailedError" in raw
            raise InstagramError(f"Instagram refused the upload (HTTP {resp.status_code} from {where}): {raw}",
                                 transient=resp.status_code >= 500 or hiccup)
        if on_container:
            on_container(container)  # only once the video is in: an empty container is useless later
        return self._finish(container)

    def _finish(self, container: str) -> tuple[str, str]:
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
        try:
            media_id = self._request("POST", f"{self.base}/{self.user_id()}/media_publish",
                                     {"creation_id": container})["id"]
        except InstagramError as exc:
            # Meta sometimes answers "unexpected error" here and publishes anyway: look before failing.
            if not exc.transient:
                raise
            self._sleep(15)
            try:
                code = self._request("GET", f"{self.base}/{container}", {"fields": "status_code"}).get("status_code")
            except InstagramError:
                raise exc
            if code != "PUBLISHED":
                raise
            log.info("Instagram: publish call failed (%s) but the reel went out", exc)
            return self._latest_media()
        try:
            link = self._request("GET", f"{self.base}/{media_id}", {"fields": "permalink"}).get("permalink", "")
        except InstagramError:
            link = ""  # posted fine; the link is a nicety
        return str(media_id), link
