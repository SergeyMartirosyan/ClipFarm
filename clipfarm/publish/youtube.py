"""YouTube Shorts via the YouTube Data API (OAuth "Desktop app" client, no Google libraries).

Login once on the VM with `python -m clipfarm.cli youtube-login`: it prints a Google link, you
allow access, and paste back the address the browser ends on. The refresh token is kept in
data/youtube_token.json. A vertical video under 3 minutes becomes a Short by itself.

Google keeps uploads from API projects it hasn't audited private, whatever we ask for, so every
upload reads back its real privacy and the group is told when it went up as Private. While the
OAuth app is in "Testing" mode, Google also ends the login after 7 days: the poster notices and
asks for `youtube-login` again.
"""

from __future__ import annotations

import base64
import hashlib
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

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
API = "https://www.googleapis.com/youtube/v3"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
SCOPES = "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly"
REDIRECT_URI = "http://127.0.0.1:8765/"
TITLE_LIMIT = 100
DESCRIPTION_LIMIT = 4900
TAGS_LIMIT = 450
QUOTA_WAIT = 6 * 3600            # quota resets at midnight Pacific; try again in a few hours
LOGIN_HINT = "run `.venv/bin/python -m clipfarm.cli youtube-login` on the VM"
QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded", "uploadLimitExceeded", "rateLimitExceeded",
                 "userRateLimitExceeded"}


# --- login -----------------------------------------------------------------------------------------

def load_client(path: Path) -> tuple[str, str]:
    """(client_id, client_secret) from the JSON downloaded in Google Cloud > Credentials."""
    if not path.exists():
        raise PublishError(f"YouTube OAuth client file not found: {path}", transient=False)
    data = json.loads(path.read_text())
    block = data.get("installed") or data.get("web") or {}
    if not block.get("client_id") or not block.get("client_secret"):
        raise PublishError(f"{path.name} has no client_id/client_secret (download it again as a Desktop app client)",
                           transient=False)
    return block["client_id"], block["client_secret"]


def new_login() -> dict:
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return {"verifier": verifier, "challenge": challenge, "state": secrets.token_urlsafe(16)}


def auth_url(client_id: str, login: dict) -> str:
    return AUTH_URL + "?" + urlencode({
        "client_id": client_id, "redirect_uri": REDIRECT_URI, "response_type": "code", "scope": SCOPES,
        "access_type": "offline", "prompt": "consent", "state": login["state"],
        "code_challenge": login["challenge"], "code_challenge_method": "S256",
    })


def code_from_paste(pasted: str, state: str) -> str:
    """The pasted address (http://127.0.0.1:8765/?state=...&code=...) or just the code."""
    pasted = pasted.strip()
    if "code=" not in pasted and "error=" not in pasted:
        if len(pasted) > 20 and " " not in pasted and "://" not in pasted:
            return pasted  # only the code
        raise PublishError("That doesn't look like the address with ?code= in it", transient=False)
    q = parse_qs(urlparse(pasted).query if "?" in pasted else pasted)
    if q.get("error"):
        raise PublishError(f"Google said: {q['error'][0]}", transient=False)
    if q.get("state") and q["state"][0] != state:
        raise PublishError("That address is from an older login attempt; use the newest link", transient=False)
    return q["code"][0]


def exchange_code(http: httpx.Client, client: tuple[str, str], code: str, verifier: str) -> dict:
    resp = http.post(TOKEN_URL, data={
        "code": code, "client_id": client[0], "client_secret": client[1], "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code", "code_verifier": verifier,
    })
    body = resp.json()
    if resp.status_code != 200 or "error" in body:
        raise PublishError(f"Google refused the login: {body.get('error_description') or body.get('error')}",
                           transient=False)
    if not body.get("refresh_token"):
        raise PublishError("Google returned no refresh token. Remove ClipFarm's access at "
                           "myaccount.google.com/permissions and log in again", transient=False)
    return body


def save_token(path: Path, token: dict, clock=time.time) -> None:
    path.write_text(json.dumps({"refresh_token": token["refresh_token"], "scope": token.get("scope", ""),
                                "created_at": clock()}, indent=1))
    os.chmod(path, 0o600)


# --- client ----------------------------------------------------------------------------------------

def _trim_tags(tags: list[str]) -> list[str]:
    out, total = [], 0
    for t in tags:
        t = t.lstrip("#").strip()
        if not t or len(t) > 100:
            continue
        cost = len(t) + (2 if " " in t else 0) + 1
        if total + cost > TAGS_LIMIT:
            break
        out.append(t)
        total += cost
    return out


def clean_title(title: str) -> str:
    title = " ".join(title.replace("<", "").replace(">", "").split()) or "Clip"
    return title if len(title) <= TITLE_LIMIT else title[: TITLE_LIMIT - 1].rstrip() + "…"


class YouTube:
    platform = "youtube"

    def __init__(self, cfg, db=None, http: httpx.Client | None = None, clock=time.time):
        self.cfg = cfg
        self.db = db
        self.http = http or httpx.Client(timeout=httpx.Timeout(120.0, read=600.0, write=600.0))
        self._clock = clock
        self._access: str | None = None
        self._access_until = 0.0

    # -- auth -------------------------------------------------------------------------------------

    def _refresh_token(self) -> str:
        path = self.cfg.youtube_token_path
        if not path.exists():
            raise PublishError(f"YouTube isn't logged in: {LOGIN_HINT}", transient=False, code="no_login")
        return json.loads(path.read_text())["refresh_token"]

    def access_token(self, force: bool = False) -> str:
        if self._access and not force and self._clock() < self._access_until - 60:
            return self._access
        client = load_client(self.cfg.youtube_client_path)
        try:
            resp = self.http.post(TOKEN_URL, data={"client_id": client[0], "client_secret": client[1],
                                                   "refresh_token": self._refresh_token(),
                                                   "grant_type": "refresh_token"})
            body = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise PublishError(f"YouTube login refresh failed: {exc.__class__.__name__}", transient=True) from exc
        if body.get("error") == "invalid_grant":
            raise PublishError(f"YouTube login expired or was revoked: {LOGIN_HINT}", transient=False,
                               code="invalid_grant")
        if resp.status_code != 200 or "access_token" not in body:
            raise PublishError(f"YouTube login refresh failed: {body.get('error_description') or body.get('error')}",
                               transient=resp.status_code >= 500)
        self._access = body["access_token"]
        self._access_until = self._clock() + float(body.get("expires_in", 3600))
        return self._access

    def maybe_refresh(self) -> bool:
        """Checks the login still works (it raises invalid_grant when it doesn't)."""
        self.access_token(force=True)
        return True

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.access_token()}"}

    # -- HTTP -------------------------------------------------------------------------------------

    def _call(self, method: str, url: str, headers: dict | None = None, **kw) -> httpx.Response:
        """One request, retried once after refreshing an expired access token."""
        for attempt in range(2):
            try:
                resp = self.http.request(method, url, headers={**(headers or {}), **self._headers()}, **kw)
            except httpx.HTTPError as exc:
                raise PublishError(f"YouTube network error: {exc.__class__.__name__}", transient=True) from exc
            if resp.status_code == 401 and attempt == 0:
                self.access_token(force=True)
                continue
            if resp.status_code >= 400:
                raise self._error(resp)
            return resp
        raise PublishError(f"YouTube keeps refusing the login: {LOGIN_HINT}", transient=False)

    @staticmethod
    def _error(resp: httpx.Response) -> PublishError:
        try:
            err = resp.json().get("error") or {}
        except ValueError:
            err = {}
        reasons = {e.get("reason") for e in err.get("errors") or [] if isinstance(e, dict)}
        msg = err.get("message") or f"HTTP {resp.status_code}"
        if reasons & QUOTA_REASONS:
            return PublishError(f"YouTube daily upload limit reached ({msg}); trying again later",
                                transient=True, code=resp.status_code, retry_after=QUOTA_WAIT)
        if "youtubeSignupRequired" in reasons:
            return PublishError("This Google account has no YouTube channel yet: create one, then log in again",
                                transient=False, code=resp.status_code)
        if resp.status_code == 403 and "insufficientPermissions" in reasons:
            return PublishError(f"YouTube login is missing the upload permission: {LOGIN_HINT}", transient=False,
                                code=resp.status_code)
        return PublishError(f"YouTube: {msg} ({', '.join(sorted(r for r in reasons if r)) or resp.status_code})",
                            transient=resp.status_code >= 500 or resp.status_code == 429, code=resp.status_code)

    # -- API --------------------------------------------------------------------------------------

    def channel(self) -> dict:
        items = self._call("GET", f"{API}/channels", params={"part": "snippet", "mine": "true"}).json().get("items")
        if not items:
            raise PublishError("This Google account has no YouTube channel yet", transient=False)
        return {"id": items[0]["id"], "title": items[0]["snippet"].get("title", "")}

    def publish(self, item: PostItem) -> Published:
        path = Path(item.local_path)
        data = path.read_bytes()
        privacy = self.cfg.youtube_privacy
        meta = {
            "snippet": {"title": clean_title(item.title), "description": item.text[:DESCRIPTION_LIMIT],
                        "tags": _trim_tags(item.hashtags), "categoryId": self.cfg.youtube_category},
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False},
        }
        init = self._call("POST", UPLOAD_URL, params={"uploadType": "resumable", "part": "snippet,status"},
                          json=meta, headers={"X-Upload-Content-Length": str(len(data)),
                                              "X-Upload-Content-Type": "video/mp4"})
        upload_url = init.headers.get("location")
        if not upload_url:
            raise PublishError("YouTube didn't return an upload address", transient=True)
        video = self._call("PUT", upload_url, content=data, headers={"Content-Type": "video/mp4"}).json()
        vid = video.get("id")
        if not vid:
            raise PublishError("YouTube accepted the upload but returned no video id", transient=False)
        got = (video.get("status") or {}).get("privacyStatus", privacy)
        note = ""
        if got != privacy:
            note = (f"⚠️ YouTube put it up as {got.capitalize()}, not {privacy.capitalize()}: Google locks uploads "
                    "from unaudited API apps. Publish it by hand in YouTube Studio until the app passes the audit.")
        return Published(id=vid, url=f"https://youtube.com/shorts/{vid}", note=note, extra={"privacy": got})
