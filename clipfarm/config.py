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
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{name} in .env must be a whole number, got {raw!r}") from None


def _int_list(name: str) -> tuple[int, ...]:
    raw = _str(name).replace(",", " ").split()
    try:
        return tuple(int(x) for x in raw)
    except ValueError:
        raise SystemExit(f"{name} in .env must be numeric Telegram user ids separated by commas, got {_str(name)!r}") from None


def _float(name: str, default: float) -> float:
    raw = _str(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class Config:
    # Telegram
    telegram_token: str
    telegram_owner_id: int
    telegram_api_base: str
    telegram_allowed_users: tuple[int, ...]
    telegram_chat_id: int

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
    part_hours: float
    scout_above_hours: float
    scout_parts: int

    vision_review: bool
    vision_candidates: int

    # Download
    ytdlp_cookies: str
    keep_last_source: bool

    # Twitch signals
    twitch_client_id: str
    twitch_client_secret: str
    twitch_chat: bool

    # Layout
    face_model_path: Path
    sound_model_path: Path
    sound_labels_path: Path
    presets_path: Path

    # Posting
    instagram_access_token: str
    instagram_api_base: str
    instagram_fb_api_base: str
    instagram_user_id: str
    youtube_client_path: Path
    youtube_privacy: str
    youtube_category: str
    posts_per_day: int
    post_window: str
    post_tz: str
    caption_suffix: str

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
    def youtube_token_path(self) -> Path:
        """Written by `clipfarm.cli youtube-login`."""
        return self.data_dir / "youtube_token.json"

    @property
    def post_platforms(self) -> list[str]:
        """Platforms ClipFarm can post to: Instagram with a token in .env, YouTube once logged in."""
        out = ["instagram"] if self.instagram_access_token else []
        if self.youtube_token_path.exists():
            out.append("youtube")
        return out

    @property
    def allowed_user_ids(self) -> set[int]:
        """Who may use the bot: the owner plus TELEGRAM_ALLOWED_USERS."""
        return {self.telegram_owner_id, *self.telegram_allowed_users} - {0}

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
        telegram_allowed_users=_int_list("TELEGRAM_ALLOWED_USERS"),
        telegram_chat_id=_int("TELEGRAM_CHAT_ID", 0),
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
        clip_min_seconds=_int("CLIP_MIN_SECONDS", 15),
        clip_max_seconds=_int("CLIP_MAX_SECONDS", 35),
        max_source_hours=_float("MAX_SOURCE_HOURS", 6.0),
        part_hours=_float("PART_HOURS", 2.0),
        scout_above_hours=_float("SCOUT_ABOVE_HOURS", 12.0),
        scout_parts=_int("SCOUT_PARTS", 2),
        vision_review=_str("VISION_REVIEW", "1").lower() not in {"0", "false", "no", "off"},
        vision_candidates=_int("VISION_CANDIDATES", 20),
        ytdlp_cookies=_str("YTDLP_COOKIES"),
        keep_last_source=_str("KEEP_LAST_SOURCE", "1").lower() not in {"0", "false", "no", "off"},
        twitch_client_id=_str("TWITCH_CLIENT_ID"),
        twitch_client_secret=_str("TWITCH_CLIENT_SECRET"),
        twitch_chat=_str("TWITCH_CHAT", "1").lower() not in {"0", "false", "no", "off"},
        face_model_path=Path(
            _str("FACE_MODEL_PATH", str(data_dir / "models" / "face_detection_yunet_2023mar.onnx"))
        ).resolve(),
        sound_model_path=Path(_str("SOUND_MODEL_PATH", str(data_dir / "models" / "yamnet.tflite"))).resolve(),
        sound_labels_path=Path(_str("SOUND_LABELS_PATH", str(data_dir / "models" / "yamnet_class_map.csv"))).resolve(),
        presets_path=Path(_str("PRESETS_PATH", "./presets.json")).resolve(),
        instagram_access_token=_str("INSTAGRAM_ACCESS_TOKEN"),
        instagram_api_base=_str("INSTAGRAM_API_BASE", "https://graph.instagram.com"),
        instagram_fb_api_base=_str("INSTAGRAM_FB_API_BASE", "https://graph.facebook.com/v21.0"),
        instagram_user_id=_str("INSTAGRAM_USER_ID"),
        youtube_client_path=Path(_str("YOUTUBE_CLIENT_FILE", str(data_dir / "youtube_client.json"))).resolve(),
        youtube_privacy=_str("YOUTUBE_PRIVACY", "public"),
        youtube_category=_str("YOUTUBE_CATEGORY", "20"),
        posts_per_day=_int("POSTS_PER_DAY", 15),
        post_window=_str("POST_WINDOW", "10:00-23:00"),
        post_tz=_str("POST_TZ", "Asia/Yerevan"),
        caption_suffix=_str("CAPTION_SUFFIX"),
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
