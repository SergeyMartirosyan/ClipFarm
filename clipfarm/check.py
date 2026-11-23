"""Self-check: is everything this machine needs present and working?

    python -m clipfarm.check

Each line is OK / WARN (works, but a feature is off) / FAIL (fix before starting).
Makes one tiny request to each configured service; nothing is posted anywhere.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

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
    try:
        from faster_whisper import WhisperModel

        WhisperModel(cfg.whisper_model, device="cpu", compute_type=cfg.whisper_compute_type, local_files_only=True)
        report("OK", f"Whisper {cfg.whisper_model}")
    except Exception:
        report("WARN", f"Whisper {cfg.whisper_model}", "not downloaded yet (downloads on first job)")


def check_telegram(cfg) -> None:
    if not cfg.telegram_token or not cfg.telegram_owner_id:
        report("FAIL", "Telegram", "set TELEGRAM_BOT_TOKEN and TELEGRAM_OWNER_ID")
        return
    try:
        via = "" if cfg.telegram_api_base == "https://api.telegram.org" else f" via {cfg.telegram_api_base}"
        r = httpx.get(f"{cfg.telegram_api_base}/bot{cfg.telegram_token}/getMe", timeout=30).json()
        if r.get("ok"):
            report("OK", "Telegram", f"bot @{r['result']['username']}, owner id {cfg.telegram_owner_id}{via}")
        elif r.get("error_code") == 429:
            report("FAIL", "Telegram", f"Telegram is rate-limiting this server{via} ({r.get('description')}). "
                   "See README: Telegram relay")
        else:
            report("FAIL", "Telegram", r.get("description", "token rejected"))
    except Exception as exc:
        report("FAIL", "Telegram", str(exc))


def check_gemini(cfg) -> None:
    if not cfg.gemini_api_key:
        report("FAIL" if not cfg.anthropic_api_key else "WARN", "Gemini", "GEMINI_API_KEY not set")
        return
    try:
        from google import genai

        from .pipeline.judge import gemini_model_chain

        # Keep a reference: the SDK closes its HTTP client when the Client object is freed.
        client = genai.Client(api_key=cfg.gemini_api_key)
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

        anthropic.Anthropic(api_key=cfg.anthropic_api_key).messages.create(
            model=cfg.fallback_model, max_tokens=16, messages=[{"role": "user", "content": "Reply with ok."}]
        )
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


def check_r2(cfg) -> None:
    if not cfg.r2_enabled:
        report("WARN", "Cloudflare R2", "not configured: clips stay on this machine only")
        return
    try:
        from .storage import R2

        R2(cfg).client.head_bucket(Bucket=cfg.r2_bucket)
        detail = f"bucket {cfg.r2_bucket}"
        if not cfg.r2_public_base_url:
            detail += " (no R2_PUBLIC_BASE_URL: Instagram posting will need it)"
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
    cfg = load_config()
    print(f"ClipFarm self-check (data in {cfg.data_dir})\n")
    check_tools()
    check_models(cfg)
    check_telegram(cfg)
    check_gemini(cfg)
    check_claude(cfg)
    check_twitch(cfg)
    check_r2(cfg)
    check_youtube_cookies(cfg)
    fails = [r for r in results if r[0] == "FAIL"]
    print()
    if fails:
        print(f"{len(fails)} problem(s) to fix before starting the services.")
        sys.exit(1)
    print("Ready. Start the services and send your bot a link.")


if __name__ == "__main__":
    main()
