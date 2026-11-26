"""TikTok Content Posting API client and login (mocked TikTok endpoints)."""

import dataclasses
import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from clipfarm import db as dbm
from clipfarm.publish import PostItem, PublishError
from clipfarm.publish import tiktok as tt
from tests.test_posting import _cfg


class TikTokApi:
    def __init__(self, statuses=("PROCESSING_UPLOAD", "SEND_TO_USER_INBOX"), init_error=None, refresh_error=None):
        self.statuses = list(statuses)
        self.init_error = init_error
        self.refresh_error = refresh_error
        self.calls = []
        self.uploaded = None
        self.init_body = None

    def __call__(self, req: httpx.Request):
        path = req.url.path
        self.calls.append(path)
        ok = {"error": {"code": "ok", "message": "", "log_id": "x"}}
        if path == "/v2/oauth/token/":
            form = parse_qs(req.content.decode())
            if form["grant_type"] == ["refresh_token"] and self.refresh_error:
                return httpx.Response(400, json={"error": self.refresh_error, "error_description": "bad"})
            return httpx.Response(200, json={"access_token": "act.new", "expires_in": 86400, "refresh_token": "rft.2",
                                             "refresh_expires_in": 31536000, "open_id": "o1", "scope": tt.SCOPES})
        if req.url.host == "upload.tiktok.example":
            self.uploaded = (dict(req.headers), req.content)
            return httpx.Response(201)
        assert req.headers["authorization"].startswith("Bearer act.")
        if path == "/v2/post/publish/creator_info/query/":
            return httpx.Response(200, json={**ok, "data": {"creator_username": "los.clipos",
                                                            "privacy_level_options": ["SELF_ONLY"]}})
        if path in ("/v2/post/publish/inbox/video/init/", "/v2/post/publish/video/init/"):
            self.init_body = (path, json.loads(req.content))
            if self.init_error:
                return httpx.Response(403, json={"error": {"code": self.init_error, "message": "no"}})
            return httpx.Response(200, json={**ok, "data": {"publish_id": "p1", "upload_url": "https://upload.tiktok.example/u1"}})
        if path == "/v2/post/publish/status/fetch/":
            status = self.statuses.pop(0) if self.statuses else "SEND_TO_USER_INBOX"
            data = {"status": status}
            if status == "PUBLISH_COMPLETE":
                data["publicaly_available_post_id"] = [7123]
            return httpx.Response(200, json={**ok, "data": data})
        return httpx.Response(404, json={"error": {"code": "not_found", "message": path}})


def _tt(tmp_path, api, expires_in=86400, **over):
    cfg = _cfg(tmp_path, tiktok_client_key="ck", tiktok_client_secret="cs", **over)
    tt.save_token(cfg.tiktok_token_path, {"access_token": "act.1", "expires_in": expires_in, "refresh_token": "rft.1"})
    d = dbm.Database(cfg.db_path)
    return tt.TikTok(cfg, d, http=httpx.Client(transport=httpx.MockTransport(api)), sleep=lambda s: None), cfg, d


def _item(tmp_path, key="post9"):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0123456789")
    return PostItem(title="t", text="Speed vs the cops #gta6", hashtags=["gta6"], local_path=video, cover_offset=2.5,
                    key=key)


def test_draft_upload(tmp_path):
    api = TikTokApi()
    client, cfg, d = _tt(tmp_path, api)
    done = client.publish(_item(tmp_path))
    assert done.id == "p1" and "draft" in done.note
    path, body = api.init_body
    assert path.endswith("/inbox/video/init/") and body["source_info"]["video_size"] == 10
    headers, content = api.uploaded
    assert content == b"0123456789" and headers["content-range"] == "bytes 0-9/10"
    assert d.kv_get("tt_publish_post9") is None


def test_direct_post_with_allowed_privacy(tmp_path):
    api = TikTokApi(statuses=("PROCESSING_DOWNLOAD", "PUBLISH_COMPLETE"))
    client, cfg, d = _tt(tmp_path, api, tiktok_mode="direct")
    done = client.publish(_item(tmp_path))
    assert done.url == "https://www.tiktok.com/@los.clipos/video/7123" and "SELF_ONLY" in done.note
    post_info = api.init_body[1]["post_info"]
    assert post_info["privacy_level"] == "SELF_ONLY" and post_info["video_cover_timestamp_ms"] == 2500


def test_privacy_not_allowed_is_explained(tmp_path):
    client, _, _ = _tt(tmp_path, TikTokApi(), tiktok_mode="direct", tiktok_privacy="PUBLIC_TO_EVERYONE")
    with pytest.raises(PublishError, match="SELF_ONLY") as e:
        client.publish(_item(tmp_path))
    assert not e.value.transient


def test_unaudited_error_is_explained(tmp_path):
    client, _, _ = _tt(tmp_path, TikTokApi(init_error="unaudited_client_can_only_post_to_private_accounts"))
    with pytest.raises(PublishError, match="private account") as e:
        client.publish(_item(tmp_path))
    assert not e.value.transient


def test_retry_checks_the_earlier_upload_first(tmp_path):
    api = TikTokApi(statuses=("SEND_TO_USER_INBOX",))
    client, cfg, d = _tt(tmp_path, api)
    d.kv_set("tt_publish_post9", "p0")
    assert client.publish(_item(tmp_path)).id == "p0"
    assert api.uploaded is None and api.init_body is None  # nothing uploaded twice


def test_token_refresh_and_expiry(tmp_path):
    api = TikTokApi()
    client, cfg, _ = _tt(tmp_path, api, expires_in=60)  # about to expire: refreshed before use
    assert client.maybe_refresh() is True
    saved = json.loads(cfg.tiktok_token_path.read_text())
    assert saved["access_token"] == "act.new" and saved["refresh_token"] == "rft.2"
    api2 = TikTokApi(refresh_error="invalid_grant")
    client2, _, _ = _tt(tmp_path / "b", api2, expires_in=60)
    with pytest.raises(PublishError, match="invalid_grant") as e:
        client2.maybe_refresh()
    assert not e.value.transient


def test_login_helpers(tmp_path):
    cfg = _cfg(tmp_path, tiktok_client_key="ck", tiktok_client_secret="cs",
               tiktok_redirect_uri="https://site.example/tiktok-callback")
    state = tt.new_state()
    q = parse_qs(urlparse(tt.auth_url(cfg, state)).query)
    assert q["client_key"] == ["ck"] and q["redirect_uri"] == ["https://site.example/tiktok-callback"]
    assert "video.upload" in q["scope"][0] and "video.publish" in q["scope"][0]
    assert tt.code_from_paste(f"https://site.example/tiktok-callback?code=C1&scopes=x&state={state}", state) == "C1"
    with pytest.raises(PublishError, match="older login"):
        tt.code_from_paste("https://site.example/tiktok-callback?code=C1&state=old", state)
    with pytest.raises(PublishError, match="isn't the address"):
        tt.code_from_paste("someone@example.com", state)


def test_tiktok_joins_the_platforms_after_login(tmp_path):
    cfg = _cfg(tmp_path, instagram_access_token="", tiktok_client_key="ck")
    assert "tiktok" not in cfg.post_platforms
    cfg.tiktok_token_path.write_text("{}")
    assert cfg.post_platforms == ["tiktok"]
