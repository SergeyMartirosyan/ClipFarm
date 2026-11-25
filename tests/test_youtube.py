"""YouTube Shorts uploader and login (mocked Google endpoints)."""

import dataclasses
import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from clipfarm.publish import PostItem, PublishError
from clipfarm.publish import youtube as yt
from tests.test_posting import _cfg


class Google:
    """A tiny fake of the Google token endpoint and the YouTube upload API."""

    def __init__(self, privacy="public", upload_error=None, refresh_error=None):
        self.privacy = privacy
        self.upload_error = upload_error
        self.refresh_error = refresh_error
        self.calls = []
        self.meta = None
        self.body = None
        self.expired_once = False

    def __call__(self, req: httpx.Request):
        url = str(req.url)
        self.calls.append((req.method, req.url.host + req.url.path))
        if url.startswith(yt.TOKEN_URL):
            form = parse_qs(req.content.decode())
            if form["grant_type"] == ["authorization_code"]:
                assert form["code"] == ["CODE"] and form["code_verifier"]
                return httpx.Response(200, json={"access_token": "A0", "refresh_token": "R1", "expires_in": 3599})
            if self.refresh_error:
                return httpx.Response(400, json={"error": self.refresh_error})
            assert form["refresh_token"] == ["R1"] and form["client_secret"] == ["sec"]
            return httpx.Response(200, json={"access_token": "A1", "expires_in": 3599})
        if req.url.path == "/youtube/v3/channels":
            return httpx.Response(200, json={"items": [{"id": "UC1", "snippet": {"title": "Klip Ktrogh"}}]})
        if req.url.path == "/upload/youtube/v3/videos" and req.method == "POST":
            assert req.headers["authorization"] == "Bearer A1"
            if self.upload_error:
                return httpx.Response(self.upload_error[0], json={"error": self.upload_error[1]})
            assert req.headers["x-upload-content-type"] == "video/mp4"
            self.meta = json.loads(req.content)
            return httpx.Response(200, headers={"Location": "https://www.googleapis.com/upload/session/42"})
        if req.url.path == "/upload/session/42":
            self.body = req.content
            return httpx.Response(200, json={"id": "vid123", "status": {"privacyStatus": self.privacy}})
        return httpx.Response(404, json={"error": {"message": "unknown " + url}})


def _yt(tmp_path, google, **cfg_over):
    cfg = _cfg(tmp_path, **cfg_over)
    cfg.youtube_client_path.write_text(json.dumps({"installed": {"client_id": "cid", "client_secret": "sec"}}))
    cfg.youtube_token_path.write_text(json.dumps({"refresh_token": "R1"}))
    return yt.YouTube(cfg, http=httpx.Client(transport=httpx.MockTransport(google))), cfg


def _item(tmp_path, **over):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"VIDEO")
    fields = dict(title="Speed <gets> chased by the cops " + "x" * 120, text="He did NOT expect that\n\n#gta6 #ishowspeed",
                  hashtags=["gta6", "#ishowspeed"], local_path=video)
    fields.update(over)
    return PostItem(**fields)


def test_upload_flow(tmp_path):
    g = Google()
    client, _ = _yt(tmp_path, g)
    done = client.publish(_item(tmp_path))
    assert done.id == "vid123" and done.url == "https://youtube.com/shorts/vid123" and done.note == ""
    snippet, status = g.meta["snippet"], g.meta["status"]
    assert len(snippet["title"]) == 100 and "<" not in snippet["title"] and snippet["title"].endswith("…")
    assert snippet["tags"] == ["gta6", "ishowspeed"] and snippet["categoryId"] == "20"
    assert snippet["description"].startswith("He did NOT expect that")
    assert status == {"privacyStatus": "public", "selfDeclaredMadeForKids": False}
    assert g.body == b"VIDEO"


def test_private_lock_is_reported(tmp_path):
    client, _ = _yt(tmp_path, Google(privacy="private"))
    done = client.publish(_item(tmp_path))
    assert "Private" in done.note and "YouTube Studio" in done.note


def test_quota_is_retried_later(tmp_path):
    client, _ = _yt(tmp_path, Google(upload_error=(403, {"message": "quota", "errors": [{"reason": "quotaExceeded"}]})))
    with pytest.raises(PublishError) as e:
        client.publish(_item(tmp_path))
    assert e.value.transient and e.value.retry_after == yt.QUOTA_WAIT


def test_bad_request_is_permanent(tmp_path):
    client, _ = _yt(tmp_path, Google(upload_error=(400, {"message": "bad title", "errors": [{"reason": "invalidTitle"}]})))
    with pytest.raises(PublishError) as e:
        client.publish(_item(tmp_path))
    assert not e.value.transient and "invalidTitle" in str(e.value)


def test_expired_login_asks_to_log_in_again(tmp_path):
    client, _ = _yt(tmp_path, Google(refresh_error="invalid_grant"))
    with pytest.raises(PublishError) as e:
        client.maybe_refresh()
    assert not e.value.transient and e.value.code == "invalid_grant" and "youtube-login" in str(e.value)


def test_access_token_is_reused(tmp_path):
    g = Google()
    client, _ = _yt(tmp_path, g)
    client.channel()
    client.channel()
    assert sum(1 for c in g.calls if c[1].endswith("/token")) == 1


def test_login_helpers(tmp_path):
    login = yt.new_login()
    url = yt.auth_url("cid", login)
    q = parse_qs(urlparse(url).query)
    assert q["access_type"] == ["offline"] and q["prompt"] == ["consent"] and q["code_challenge_method"] == ["S256"]
    assert "youtube.upload" in q["scope"][0] and q["redirect_uri"] == [yt.REDIRECT_URI]
    pasted = f"http://127.0.0.1:8765/?state={login['state']}&code=CODE&scope=x"
    assert yt.code_from_paste(pasted, login["state"]) == "CODE"
    with pytest.raises(PublishError, match="older login"):
        yt.code_from_paste("http://127.0.0.1:8765/?state=old&code=CODE", login["state"])
    with pytest.raises(PublishError, match="access_denied"):
        yt.code_from_paste("http://127.0.0.1:8765/?error=access_denied", login["state"])
    for bad in ("hello", "someone.support@gmail.com"):
        with pytest.raises(PublishError, match="isn't the address"):
            yt.code_from_paste(bad, login["state"])
    assert yt.code_from_paste("4/0AUn6Hd9ELOjHm-Y-xvssadGE15tD7dzn", login["state"]).startswith("4/")
    with httpx.Client(transport=httpx.MockTransport(Google())) as http:
        token = yt.exchange_code(http, ("cid", "sec"), "CODE", login["verifier"])
    path = tmp_path / "tok.json"
    yt.save_token(path, token)
    assert json.loads(path.read_text())["refresh_token"] == "R1" and oct(path.stat().st_mode)[-3:] == "600"


def test_poster_builds_youtube_after_login(tmp_path):
    from clipfarm.poster import Poster
    from tests.test_posting import FakeTg

    client, cfg = _yt(tmp_path, Google(), instagram_access_token="")
    p = Poster(cfg=cfg, tg=FakeTg())
    assert isinstance(p.publisher("youtube"), yt.YouTube) and p.publisher("instagram") is None
