"""TikTok via the Content Posting API.

Login once on the VM with `python -m clipfarm.cli tiktok-login`: it prints a TikTok link, you allow
access, TikTok sends the browser to the site's /tiktok-callback page, and you paste that address
back. The access token lives 24 h and is refreshed with the refresh token (valid a year), both kept
in data/tiktok_token.json.

Until TikTok audits the app, it only allows private posts (and only to private accounts), so the
default mode is "draft": the video lands in the TikTok app's inbox and you publish it there with a
tap. TIKTOK_MODE=direct posts straight away with TIKTOK_PRIVACY (SELF_ONLY until the audit, then
PUBLIC_TO_EVERYONE). Retries are duplicate-proof: an attempt's publish_id is remembered and its
status checked before anything is uploaded again.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from . import PostItem, PublishError, Published

log = logging.getLogger(__name__)

AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
API = "https://open.tiktokapis.com/v2"
TOKEN_URL = f"{API}/oauth/token/"
SCOPES = "user.info.basic,video.upload,video.publish"
LOGIN_HINT = "run `.venv/bin/python -m clipfarm.cli tiktok-login` on the VM"
CAPTION_LIMIT = 2200
POLL_INTERVAL = 10
POLL_TIMEOUT = 900
REFRESH_MARGIN = 2 * 3600
TRANSIENT = {"rate_limit_exceeded", "internal_error", "spam_risk_too_many_pending_share"}
EXPLAIN = {
    "unaudited_client_can_only_post_to_private_accounts":
        "TikTok only lets an unaudited app post to a private account: make the TikTok account private, "
        "or use TIKTOK_MODE=draft (the default)",
    "privacy_level_option_mismatch": "TIKTOK_PRIVACY isn't allowed for this account (unaudited apps: SELF_ONLY)",
    "spam_risk_too_many_posts": "TikTok's daily post limit for this account is reached",
    "spam_risk_user_banned_from_posting": "TikTok has blocked posting for this account",
    "scope_not_authorized": f"the login is missing a permission (video.upload / video.publish): {LOGIN_HINT}",
    "access_token_invalid": f"the TikTok login isn't valid any more: {LOGIN_HINT}",
}


# --- login -----------------------------------------------------------------------------------------

def auth_url(cfg, state: str) -> str:
    return AUTH_URL + "?" + urlencode({
        "client_key": cfg.tiktok_client_key, "scope": SCOPES, "response_type": "code",
        "redirect_uri": cfg.tiktok_redirect_uri, "state": state,
    })


def new_state() -> str:
    return secrets.token_urlsafe(16)


def code_from_paste(pasted: str, state: str) -> str:
    pasted = pasted.strip()
    if "code=" not in pasted and "error=" not in pasted:
        raise PublishError("That isn't the address: it ends in /tiktok-callback?code=...", transient=False)
    q = parse_qs(urlparse(pasted).query if "?" in pasted else pasted)
    if q.get("error"):
        raise PublishError(f"TikTok said: {(q.get('error_description') or q['error'])[0]}", transient=False)
    if q.get("state") and q["state"][0] != state:
        raise PublishError("That address is from an older login attempt; use the newest link", transient=False)
    return q["code"][0]


def _token_request(http: httpx.Client, form: dict) -> dict:
    try:
        resp = http.post(TOKEN_URL, data=form, headers={"Content-Type": "application/x-www-form-urlencoded"})
        body = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise PublishError(f"TikTok login request failed: {exc.__class__.__name__}", transient=True) from exc
    if "access_token" not in body:
        err = body.get("error") or "unknown"
        desc = body.get("error_description") or ""
        raise PublishError(f"TikTok refused the login: {err} {desc}".strip(), transient=False, code=err)
    return body


def exchange_code(http: httpx.Client, cfg, code: str) -> dict:
    return _token_request(http, {
        "client_key": cfg.tiktok_client_key, "client_secret": cfg.tiktok_client_secret, "code": code,
        "grant_type": "authorization_code", "redirect_uri": cfg.tiktok_redirect_uri,
    })


def save_token(path: Path, token: dict, clock=time.time) -> None:
    now = clock()
    path.write_text(json.dumps({
        "access_token": token["access_token"], "expires_at": now + float(token.get("expires_in", 86400)),
        "refresh_token": token["refresh_token"],
        "refresh_expires_at": now + float(token.get("refresh_expires_in", 365 * 86400)),
        "open_id": token.get("open_id", ""), "scope": token.get("scope", ""),
    }, indent=1))
    os.chmod(path, 0o600)


# --- client ----------------------------------------------------------------------------------------

class TikTok:
    platform = "tiktok"

    def __init__(self, cfg, db=None, http: httpx.Client | None = None, sleep=time.sleep, clock=time.time):
        self.cfg = cfg
        self.db = db
        self.http = http or httpx.Client(timeout=httpx.Timeout(60.0, read=600.0, write=600.0))
        self._sleep = sleep
        self._clock = clock
        self._username: str | None = None

    # -- auth -------------------------------------------------------------------------------------

    def _load(self) -> dict:
        path = self.cfg.tiktok_token_path
        if not path.exists():
            raise PublishError(f"TikTok isn't logged in: {LOGIN_HINT}", transient=False, code="no_login")
        return json.loads(path.read_text())

    def access_token(self, force: bool = False) -> str:
        tok = self._load()
        if not force and self._clock() < tok["expires_at"] - REFRESH_MARGIN:
            return tok["access_token"]
        if self._clock() > tok.get("refresh_expires_at", 0):
            raise PublishError(f"TikTok login expired: {LOGIN_HINT}", transient=False, code="invalid_grant")
        fresh = _token_request(self.http, {
            "client_key": self.cfg.tiktok_client_key, "client_secret": self.cfg.tiktok_client_secret,
            "grant_type": "refresh_token", "refresh_token": tok["refresh_token"],
        })
        fresh.setdefault("refresh_token", tok["refresh_token"])
        save_token(self.cfg.tiktok_token_path, fresh, self._clock)
        return fresh["access_token"]

    def maybe_refresh(self) -> bool:
        tok = self._load()
        if self._clock() < tok["expires_at"] - REFRESH_MARGIN:
            return False
        self.access_token(force=True)
        return True

    # -- HTTP -------------------------------------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = self.http.post(f"{API}{path}", json=payload, headers={
                "Authorization": f"Bearer {self.access_token()}", "Content-Type": "application/json; charset=UTF-8"})
        except httpx.HTTPError as exc:
            raise PublishError(f"TikTok network error: {exc.__class__.__name__}", transient=True) from exc
        try:
            body = resp.json()
        except ValueError as exc:
            raise PublishError(f"TikTok HTTP {resp.status_code}, not JSON", transient=resp.status_code >= 500) from exc
        err = body.get("error") or {}
        code = err.get("code", "ok")
        if code != "ok" or resp.status_code >= 400:
            msg = EXPLAIN.get(code) or err.get("message") or f"HTTP {resp.status_code}"
            raise PublishError(f"TikTok: {msg} ({code})", transient=code in TRANSIENT or resp.status_code >= 500,
                               code=code)
        return body.get("data") or {}

    # -- API --------------------------------------------------------------------------------------

    def creator(self) -> dict:
        data = self._post("/post/publish/creator_info/query/", {})
        self._username = data.get("creator_username") or self._username
        return data

    def publish(self, item: PostItem) -> Published:
        slot = f"tt_publish_{item.key}" if item.key else None
        draft = self.cfg.tiktok_mode != "direct"
        saved = self.db.kv_get(slot) if (slot and self.db) else None
        try:
            if saved:
                status = self._status(saved)
                if status in ("FAILED", ""):
                    saved = None
            if not saved:
                saved = self._upload(Path(item.local_path), item, draft)
                if slot and self.db:
                    self.db.kv_set(slot, saved)
            done = self._wait(saved, draft)
        except PublishError as exc:
            if slot and self.db and not exc.transient:
                self.db.kv_set(slot, None)
            raise
        if slot and self.db:
            self.db.kv_set(slot, None)
        return done

    def _upload(self, path: Path, item: PostItem, draft: bool) -> str:
        size = path.stat().st_size
        source = {"source": "FILE_UPLOAD", "video_size": size, "chunk_size": size, "total_chunk_count": 1}
        if draft:
            data = self._post("/post/publish/inbox/video/init/", {"source_info": source})
        else:
            creator = self.creator()
            privacy = self.cfg.tiktok_privacy
            allowed = creator.get("privacy_level_options") or []
            if allowed and privacy not in allowed:
                raise PublishError(f"TikTok allows only {', '.join(allowed)} for this account, not {privacy} "
                                   "(set TIKTOK_PRIVACY)", transient=False)
            post_info = {"title": item.text[:CAPTION_LIMIT], "privacy_level": privacy,
                         "disable_duet": False, "disable_comment": False, "disable_stitch": False}
            if item.cover_offset is not None:
                post_info["video_cover_timestamp_ms"] = int(item.cover_offset * 1000)
            data = self._post("/post/publish/video/init/", {"post_info": post_info, "source_info": source})
        publish_id, upload_url = data.get("publish_id"), data.get("upload_url")
        if not publish_id or not upload_url:
            raise PublishError("TikTok didn't return an upload address", transient=True)
        try:
            resp = self.http.put(upload_url, content=path.read_bytes(), headers={
                "Content-Type": "video/mp4", "Content-Range": f"bytes 0-{size - 1}/{size}"})
        except httpx.HTTPError as exc:
            raise PublishError(f"TikTok upload network error: {exc.__class__.__name__}", transient=True) from exc
        if resp.status_code >= 400:
            raise PublishError(f"TikTok refused the upload: HTTP {resp.status_code} {resp.text[:200]}",
                               transient=resp.status_code >= 500)
        return publish_id

    def _status(self, publish_id: str) -> str:
        try:
            data = self._post("/post/publish/status/fetch/", {"publish_id": publish_id})
        except PublishError as exc:
            if exc.transient:
                raise
            return ""
        self._last = data
        return data.get("status", "")

    def _wait(self, publish_id: str, draft: bool) -> Published:
        deadline = self._clock() + POLL_TIMEOUT
        while True:
            status = self._status(publish_id)
            if status == "FAILED":
                reason = (getattr(self, "_last", {}) or {}).get("fail_reason", "unknown")
                raise PublishError(f"TikTok couldn't process the video: {reason}", transient=False)
            if status == "SEND_TO_USER_INBOX" or (draft and status == "PUBLISH_COMPLETE"):
                return Published(id=publish_id, note="📥 Sent to the TikTok app as a draft: open TikTok, tap the "
                                                     "notification and post it (add a sound there if you like).")
            if status == "PUBLISH_COMPLETE":
                ids = (getattr(self, "_last", {}) or {}).get("publicaly_available_post_id") or []
                url = ""
                if ids:
                    user = self._username or self.creator().get("creator_username", "")
                    url = f"https://www.tiktok.com/@{user}/video/{ids[0]}" if user else ""
                note = "" if self.cfg.tiktok_privacy == "PUBLIC_TO_EVERYONE" else \
                    f"Posted as {self.cfg.tiktok_privacy} (TikTok's rule until the app is audited)."
                return Published(id=publish_id, url=url, note=note)
            if self._clock() > deadline:
                raise PublishError("TikTok is still processing the video; checking again later", transient=True)
            self._sleep(POLL_INTERVAL)
