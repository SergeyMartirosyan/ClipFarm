"""Settings, read once from the environment (and .env if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _str(name: str, default: str = "") -> str:
    """An empty value in .env (e.g. `GEMINI_MODEL=`) means 'use the default'."""
    return os.environ.get(name, "").strip() or default


def _int(name: str, default: int) -> int:
    raw = _str(name)
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = _str(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class Config:
    # Telegram
    telegram_token: str
    telegram_owner_id: int
    telegram_api_base: str

    # Paths
    data_dir: Path
    local_retention_days: int

    # Judge
    gemini_api_key: str
    gemini_model: str
    anthropic_api_key: str
    fallback_model: str
    fallback_monthly_cap_usd: float

    # Whisper
    whisper_model: str
    whisper_compute_type: str
    whisper_threads: int

    # Selection
    clips_per_video: int
    clip_min_seconds: int
    clip_max_seconds: int
    max_source_hours: float

    vision_review: bool
    vision_candidates: int

    # Download
    ytdlp_cookies: str

    # Twitch signals
    twitch_client_id: str
    twitch_client_secret: str
    twitch_chat: bool

    # Layout
    face_model_path: Path
    presets_path: Path

    # R2
    r2_account_id: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket: str
    r2_public_base_url: str
    r2_retention_days: int

    @property
    def db_path(self) -> Path:
        return self.data_dir / "clipfarm.db"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def clips_dir(self) -> Path:
        return self.data_dir / "clips"

    @property
    def r2_enabled(self) -> bool:
        return all(
            [self.r2_account_id, self.r2_access_key_id, self.r2_secret_access_key, self.r2_bucket]
        )

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_token and self.telegram_owner_id)


def load_config(env_file: str | None = None) -> Config:
    # .env wins over leftovers in the shell (e.g. after `set -a; . ./.env`), so edits take effect.
    load_dotenv(env_file or ".env", override=True)
    data_dir = Path(_str("DATA_DIR", "./data")).resolve()
    cfg = Config(
        telegram_token=_str("TELEGRAM_BOT_TOKEN"),
        telegram_owner_id=_int("TELEGRAM_OWNER_ID", 0),
        telegram_api_base=_str("TELEGRAM_API_BASE", "https://api.telegram.org").rstrip("/"),
        data_dir=data_dir,
        local_retention_days=_int("LOCAL_RETENTION_DAYS", 10),
        gemini_api_key=_str("GEMINI_API_KEY"),
        gemini_model=_str("GEMINI_MODEL", "gemini-flash-latest"),
        anthropic_api_key=_str("ANTHROPIC_API_KEY"),
        fallback_model=_str("FALLBACK_MODEL", "claude-haiku-5-5"),
        fallback_monthly_cap_usd=_float("FALLBACK_MONTHLY_CAP_USD", 2.0),
        whisper_model=_str("WHISPER_MODEL", "small.en"),
        whisper_compute_type=_str("WHISPER_COMPUTE_TYPE", "int8"),
        whisper_threads=_int("WHISPER_THREADS", os.cpu_count() or 2),
        clips_per_video=_int("CLIPS_PER_VIDEO", 12),
        clip_min_seconds=_int("CLIP_MIN_SECONDS", 20),
        clip_max_seconds=_int("CLIP_MAX_SECONDS", 60),
        max_source_hours=_float("MAX_SOURCE_HOURS", 6.0),
        vision_review=_str("VISION_REVIEW", "1").lower() not in {"0", "false", "no", "off"},
        vision_candidates=_int("VISION_CANDIDATES", 20),
        ytdlp_cookies=_str("YTDLP_COOKIES"),
        twitch_client_id=_str("TWITCH_CLIENT_ID"),
        twitch_client_secret=_str("TWITCH_CLIENT_SECRET"),
        twitch_chat=_str("TWITCH_CHAT", "1").lower() not in {"0", "false", "no", "off"},
        face_model_path=Path(
            _str("FACE_MODEL_PATH", str(data_dir / "models" / "face_detection_yunet_2023mar.onnx"))
        ).resolve(),
        presets_path=Path(_str("PRESETS_PATH", "./presets.json")).resolve(),
        r2_account_id=_str("R2_ACCOUNT_ID"),
        r2_access_key_id=_str("R2_ACCESS_KEY_ID"),
        r2_secret_access_key=_str("R2_SECRET_ACCESS_KEY"),
        r2_bucket=_str("R2_BUCKET"),
        r2_public_base_url=_str("R2_PUBLIC_BASE_URL"),
        r2_retention_days=_int("R2_RETENTION_DAYS", 10),
    )
    for d in (cfg.data_dir, cfg.work_dir, cfg.clips_dir):
        d.mkdir(parents=True, exist_ok=True)
    return cfg
