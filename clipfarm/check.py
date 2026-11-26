"""Self-check: is everything this machine needs present and working?

    python -m clipfarm.check                 # everything
    python -m clipfarm.check --no-telegram   # skip Telegram (while it is rate-limiting the bot)
    python -m clipfarm.check --deep          # also run Whisper for real on a 3 s test sound (~30 s)

Each line is OK / WARN (works, but a feature is off) / FAIL (fix before starting).
What it sends: Telegram getMe; Gemini: list models + one tiny prompt (free tier); Claude: a model
lookup (free, no tokens); Twitch: an app token request; R2: a bucket check. Nothing is posted.

Don't run it in a loop while Telegram reports 429: every getMe after the countdown ends makes
Telegram try to log the bot in again.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .config import load_config

results: list[tuple[str, str, str]] = []


def report(level: str, name: str, detail: str = "") -> None:
    results.append((level, name, detail))
    mark = {"OK": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[level]
    print(f"{mark} {name}" + (f": {detail}" if detail else ""), flush=True)


def check_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        report("OK" if shutil.which(tool) else "FAIL", tool, "" if shutil.which(tool) else "not installed")
    if shutil.which("ffmpeg"):
        filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
        ok = " ass " in filters
        report("OK" if ok else "FAIL", "ffmpeg captions (libass)", "" if ok else "this ffmpeg can't burn captions")
    fonts = subprocess.run(["fc-list"], capture_output=True, text=True).stdout if shutil.which("fc-list") else ""
    report("OK" if "DejaVu" in fonts else "WARN", "caption font", "" if "DejaVu" in fonts else "install fonts-dejavu-core")


def check_models(cfg) -> None:
    ok = cfg.face_model_path.exists()
    report("OK" if ok else "WARN", "face model", str(cfg.face_model_path) if ok else "missing: run scripts/fetch_models.sh (blur layout only)")
    from .pipeline.sounds import load_tagger

    if load_tagger(cfg.sound_model_path, cfg.sound_labels_path) is not None:
        report("OK", "sound model", "game sounds / laughing / screaming detection on")
    else:
        report("WARN", "sound model", "missing: run scripts/fetch_models.sh and pip install -r requirements.txt (one signal fewer)")
    try:
        from faster_whisper.utils import download_model

        download_model(cfg.whisper_model, local_files_only=True)  # finds the files, doesn't load them
        report("OK", f"Whisper {cfg.whisper_model}")
    except Exception:
        report("WARN", f"Whisper {cfg.whisper_model}", "not downloaded yet (downloads on first job)")


TOKEN_RE = re.compile(r"^\d+:[A-Za-z0-9_-]{30,}$")
LOCKED_OUT_ADVICE = (
    "This is about the bot's login, not this server or network (a relay won't help). Stop the bot "
    "(sudo systemctl stop clipfarm-bot clipfarm-worker), don't test Telegram for a few hours, then run "
    "this check once. If it persists: ask @SpamBot whether your account is limited, report it to "
    "@BotSupport, or create the bot from another Telegram account."
)


def check_whisper_runs(cfg) -> None:
    """Load the real model and transcribe a generated 3 s sound: catches library mismatches in
    seconds instead of hours into a job."""
    import tempfile

    from .media import extract_audio
    from .pipeline.transcribe import Transcriber

    started = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "tone.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=black:s=320x240:d=3",
                 "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-shortest", str(src)],
                check=True,
            )
            wav = extract_audio(src, Path(tmp) / "tone.wav")
            Transcriber(cfg.whisper_model, cfg.whisper_compute_type, cfg.whisper_threads).transcribe(wav, words=True)
        report("OK", "Whisper runs", f"{cfg.whisper_model} transcribed a test sound in {time.time() - started:.0f}s")
    except Exception as exc:
        report("FAIL", "Whisper runs", f"{exc.__class__.__name__}: {str(exc)[:200]}")


def check_telegram(cfg) -> None:
    if not cfg.telegram_token or not cfg.telegram_owner_id:
        report("FAIL", "Telegram", "set TELEGRAM_BOT_TOKEN and TELEGRAM_OWNER_ID")
        return
    bot_id = cfg.telegram_token.split(":", 1)[0]
    route = "" if cfg.telegram_api_base == "https://api.telegram.org" else f", via {cfg.telegram_api_base}"
    who = f"bot id {bot_id}{route}"  # the part before ':' is public; the secret part is never printed
    if not TOKEN_RE.match(cfg.telegram_token):
        report("FAIL", "Telegram", f"{who}: token doesn't look like a bot token (expected 123456:ABC…). Check for spaces/quotes in .env")
        return
    started = time.time()
    try:
        resp = httpx.get(f"{cfg.telegram_api_base}/bot{cfg.telegram_token}/getMe", timeout=120)
    except httpx.TimeoutException:
        report("FAIL", "Telegram", f"{who}: no answer in {time.time() - started:.0f}s. Telegram does this while it "
               "tries to log a bot in, usually followed by 429. " + LOCKED_OUT_ADVICE)
        return
    except httpx.HTTPError as exc:
        report("FAIL", "Telegram", f"{who}: can't connect ({exc.__class__.__name__}: {exc})")
        return
    took = time.time() - started
    try:
        body = resp.json()
    except ValueError:
        report("FAIL", "Telegram", f"{who}: HTTP {resp.status_code} that isn't from Telegram "
               f"({resp.text[:80]!r}). Check TELEGRAM_API_BASE.")
        return
    if body.get("ok"):
        report("OK", "Telegram", f"bot @{body['result']['username']} ({who}), owner id {cfg.telegram_owner_id}, {took:.1f}s")
        if cfg.telegram_chat_id > 0:
            report("FAIL", "Telegram group", f"TELEGRAM_CHAT_ID={cfg.telegram_chat_id} is missing the minus sign: "
                   f"group ids are negative (-{cfg.telegram_chat_id})")
        elif cfg.telegram_chat_id:
            report("OK", "Telegram group", f"bot works only in group {cfg.telegram_chat_id}; "
                   f"{len(cfg.allowed_user_ids)} people allowed")
    elif body.get("error_code") == 429:
        retry = (body.get("parameters") or {}).get("retry_after")
        report("FAIL", "Telegram", f"{who}: Telegram refuses to log this bot in right now (429, retry after {retry}s, "
               f"answered in {took:.0f}s). " + LOCKED_OUT_ADVICE)
    elif body.get("error_code") == 401:
        report("FAIL", "Telegram", f"{who}: token rejected (401). Was it revoked, or is an old token still exported "
               "in this shell? Compare the bot id with the one BotFather shows.")
    elif body.get("error_code") == 404:
        report("FAIL", "Telegram", f"{who}: 404 Not Found: malformed token or wrong TELEGRAM_API_BASE path.")
    else:
        report("FAIL", "Telegram", f"{who}: {body.get('error_code')} {body.get('description')}")


def check_gemini(cfg) -> None:
    if not cfg.gemini_api_key:
        report("FAIL" if not cfg.anthropic_api_key else "WARN", "Gemini", "GEMINI_API_KEY not set")
        return
    try:
        from google import genai

        from .pipeline.judge import gemini_model_chain

        # Keep a reference: the SDK closes its HTTP client when the Client object is freed.
        from google.genai import types

        client = genai.Client(api_key=cfg.gemini_api_key, http_options=types.HttpOptions(timeout=60_000))
        chain = gemini_model_chain(client, cfg.gemini_model)
    except Exception as exc:
        report("FAIL", "Gemini", f"key rejected or Google unreachable: {str(exc)[:200]}")
        return
    busy = []
    for model in chain:
        try:
            resp = client.models.generate_content(model=model, contents="Reply with the word ok.")
        except Exception as exc:
            busy.append(f"{model}: {str(exc)[:80]}")
            continue
        note = "" if model == cfg.gemini_model else f" ({cfg.gemini_model} unavailable, auto-picked)"
        report("OK", "Gemini", f"{model}{note} answered: {(resp.text or '').strip()[:20]}. Order: {', '.join(chain)}")
        return
    if any("503" in b or "UNAVAILABLE" in b or "429" in b for b in busy):
        report("WARN", "Gemini", "key works but every model is busy right now (temporary): " + "; ".join(busy))
    else:
        report("FAIL", "Gemini", "; ".join(busy))


def check_claude(cfg) -> None:
    if not cfg.anthropic_api_key:
        report("WARN", "Claude fallback", "ANTHROPIC_API_KEY not set (no fallback when Gemini fails)")
        return
    try:
        import anthropic

        # A model lookup checks the key and the model name without spending tokens.
        anthropic.Anthropic(api_key=cfg.anthropic_api_key, timeout=30).models.retrieve(cfg.fallback_model)
        report("OK", "Claude fallback", f"{cfg.fallback_model}, cap ${cfg.fallback_monthly_cap_usd:.2f}/month")
    except Exception as exc:
        report("FAIL", "Claude fallback", str(exc)[:200])


def check_twitch(cfg) -> None:
    if cfg.twitch_client_id and cfg.twitch_client_secret:
        try:
            from .pipeline.twitch import Helix

            Helix(cfg.twitch_client_id, cfg.twitch_client_secret)._auth()
            report("OK", "Twitch viewer clips", "app credentials accepted")
        except Exception as exc:
            report("FAIL", "Twitch viewer clips", str(exc)[:200])
    else:
        report("WARN", "Twitch viewer clips", "TWITCH_CLIENT_ID/SECRET not set (Twitch picks will be weaker)")
    report("OK" if cfg.twitch_chat else "WARN", "Twitch chat replay", "on" if cfg.twitch_chat else "off (TWITCH_CHAT=0)")


def check_instagram(cfg) -> None:
    if not cfg.instagram_access_token:
        report("WARN", "Instagram", "INSTAGRAM_ACCESS_TOKEN not set: approved clips won't be posted there")
        return
    try:
        from . import db as dbm
        from .publish.instagram import Instagram

        ig = Instagram(cfg, dbm.Database(cfg.db_path))
        me = ig.me()
        detail = f"@{me.get('username')} (user id {me.get('user_id')})"
        try:
            lim = ig.publishing_limit()
            used = lim.get("quota_usage")
            total = (lim.get("config") or {}).get("quota_total")
            if used is not None and total:
                detail += f", posts used today: {used}/{total}"
        except Exception:
            pass
        how = "via R2 URL" if (cfg.r2_enabled and cfg.r2_public_base_url) else "direct upload"
        if ig.facebook_token:
            how += ", EAA token via graph.facebook.com (not auto-renewed: the group is told when it expires)"
        report("OK", "Instagram", f"{detail}, {how}")
    except Exception as exc:
        report("FAIL", "Instagram", str(exc)[:250])


def check_tiktok(cfg) -> None:
    if not cfg.tiktok_client_key:
        report("WARN", "TikTok", "TIKTOK_CLIENT_KEY not set: no TikTok posting")
        return
    if not cfg.tiktok_token_path.exists():
        report("WARN", "TikTok", "not logged in: run `.venv/bin/python -m clipfarm.cli tiktok-login`")
        return
    try:
        from .publish.tiktok import TikTok

        info = TikTok(cfg).creator()
        mode = "drafts to the app inbox" if cfg.tiktok_mode != "direct" else f"direct, {cfg.tiktok_privacy}"
        report("OK", "TikTok", f"@{info.get('creator_username')}, {mode}; allowed: "
               + ", ".join(info.get("privacy_level_options") or []))
    except Exception as exc:
        report("FAIL", "TikTok", str(exc)[:250])


def check_cover(cfg) -> None:
    from pathlib import Path

    if Path(cfg.cover_image).exists():
        report("OK", "Cover image", f"{cfg.cover_image} (YouTube thumbnail)")
    else:
        report("WARN", "Cover image", "none: clips use a frame from the video (make-cover sets one)")
    if not cfg.cover_image_url:
        return
    try:
        import httpx

        resp = httpx.get(cfg.cover_image_url, follow_redirects=True, timeout=30)
        kind = resp.headers.get("content-type", "")
        if resp.status_code == 200 and kind.startswith("image/"):
            report("OK", "Instagram cover link", f"{cfg.cover_image_url} ({len(resp.content) // 1024} KB, {kind})")
        else:
            report("FAIL", "Instagram cover link", f"{cfg.cover_image_url} gave HTTP {resp.status_code} {kind}: "
                   "Instagram posts would fail. Fix the link or empty COVER_IMAGE_URL")
    except Exception as exc:
        report("FAIL", "Instagram cover link", f"{cfg.cover_image_url}: {exc.__class__.__name__}")


def check_youtube(cfg) -> None:
    if not cfg.youtube_client_path.exists():
        report("WARN", "YouTube", f"no OAuth client file at {cfg.youtube_client_path}: no YouTube posting")
        return
    if not cfg.youtube_token_path.exists():
        report("WARN", "YouTube", "not logged in: run `.venv/bin/python -m clipfarm.cli youtube-login`")
        return
    try:
        from .publish.youtube import YouTube

        ch = YouTube(cfg).channel()
        report("OK", "YouTube", f"channel {ch['title']}, uploads as {cfg.youtube_privacy}")
    except Exception as exc:
        report("FAIL", "YouTube", str(exc)[:250])


def check_r2(cfg) -> None:
    if not cfg.r2_enabled:
        report("WARN", "Cloudflare R2", "not configured: clips stay on this machine only")
        return
    try:
        from .storage import R2

        R2(cfg).client.head_bucket(Bucket=cfg.r2_bucket)
        detail = f"bucket {cfg.r2_bucket}"
        if not cfg.r2_public_base_url:
            detail += " (no R2_PUBLIC_BASE_URL: Instagram gets the file directly instead)"
        report("OK", "Cloudflare R2", detail)
    except Exception as exc:
        report("FAIL", "Cloudflare R2", str(exc)[:200])


def check_youtube_cookies(cfg) -> None:
    if not cfg.ytdlp_cookies:
        report("WARN", "YouTube cookies", "not set: YouTube may block downloads on cloud IPs (Twitch is fine)")
    else:
        from pathlib import Path

        ok = Path(cfg.ytdlp_cookies).exists()
        report("OK" if ok else "FAIL", "YouTube cookies", cfg.ytdlp_cookies if ok else f"file not found: {cfg.ytdlp_cookies}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="clipfarm.check")
    parser.add_argument("--no-telegram", action="store_true", help="skip the Telegram check")
    parser.add_argument("--deep", action="store_true", help="also run Whisper for real on a test sound")
    args = parser.parse_args()
    env = Path.cwd() / ".env"
    if not env.exists():
        print(f"⚠️  No .env in {Path.cwd()}: run this from the ClipFarm folder. Using environment variables only.\n")
    cfg = load_config()
    print(f"ClipFarm self-check (settings: {env if env.exists() else 'environment'}, data in {cfg.data_dir})\n")
    check_tools()
    check_models(cfg)
    if args.deep:
        check_whisper_runs(cfg)
    if args.no_telegram:
        report("WARN", "Telegram", "skipped (--no-telegram)")
    else:
        check_telegram(cfg)
    check_gemini(cfg)
    check_claude(cfg)
    check_twitch(cfg)
    check_r2(cfg)
    check_instagram(cfg)
    check_youtube(cfg)
    check_tiktok(cfg)
    check_cover(cfg)
    check_youtube_cookies(cfg)
    fails = [r for r in results if r[0] == "FAIL"]
    print()
    if fails:
        print(f"{len(fails)} problem(s) to fix before starting the services.")
        sys.exit(1)
    if args.no_telegram:
        print("Everything checked is ready. Telegram was not checked (--no-telegram).")
    else:
        print("Ready. Start the services and send your bot a link.")


if __name__ == "__main__":
    main()
