"""SQLite state shared by the bot and the worker (separate processes, WAL mode)."""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    url               TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'queued',
    error             TEXT,
    source_title      TEXT,
    uploader          TEXT,
    duration          REAL,
    chat_id           INTEGER,
    status_message_id INTEGER,
    attempts          INTEGER NOT NULL DEFAULT 0,
    user_id           INTEGER,
    user_name         TEXT NOT NULL DEFAULT '',
    range_start       REAL,
    range_end         REAL,
    part_label        TEXT NOT NULL DEFAULT '',
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS clips (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id              INTEGER NOT NULL REFERENCES jobs(id),
    idx                 INTEGER NOT NULL,
    total               INTEGER NOT NULL DEFAULT 0,
    start               REAL NOT NULL,
    end                 REAL NOT NULL,
    title               TEXT NOT NULL,
    hashtags            TEXT NOT NULL DEFAULT '',
    reason              TEXT NOT NULL DEFAULT '',
    vision_note         TEXT NOT NULL DEFAULT '',
    llm_score           REAL NOT NULL DEFAULT 0,
    signal_score        REAL NOT NULL DEFAULT 0,
    final_score         REAL NOT NULL DEFAULT 0,
    judge_model         TEXT NOT NULL DEFAULT '',
    layout              TEXT NOT NULL DEFAULT '',
    local_path          TEXT,
    r2_key              TEXT,
    public_url          TEXT,
    status              TEXT NOT NULL DEFAULT 'pending_review',
    telegram_message_id INTEGER,
    deliver_attempts    INTEGER NOT NULL DEFAULT 0,
    decided_by          TEXT NOT NULL DEFAULT '',
    overlay_text        TEXT NOT NULL DEFAULT '',
    caption             TEXT NOT NULL DEFAULT '',
    cover_offset        REAL,
    rank                INTEGER,
    created_at          REAL NOT NULL,
    decided_at          REAL
);

CREATE TABLE IF NOT EXISTS llm_usage (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL NOT NULL,
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS posts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id      INTEGER NOT NULL REFERENCES clips(id),
    platform     TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'scheduled',
    scheduled_at REAL NOT NULL,
    attempts     INTEGER NOT NULL DEFAULT 0,
    next_try_at  REAL,
    external_id  TEXT,
    url          TEXT,
    error        TEXT,
    created_at   REAL NOT NULL,
    posted_at    REAL,
    UNIQUE (clip_id, platform)
);

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_clips_status ON clips(status);
CREATE INDEX IF NOT EXISTS idx_posts_due ON posts(status, scheduled_at);
"""

# Job statuses, in pipeline order. 'new' = being announced by the bot, not yet claimable.
NEW = "new"
QUEUED = "queued"
MAX_ATTEMPTS = 2  # a job that kills the worker twice (e.g. out of memory) is failed, not retried forever
ACTIVE_STATUSES = ("downloading", "transcribing", "analyzing", "rendering")
DONE = "done"
FAILED = "failed"

# Clip statuses.
PENDING_REVIEW = "pending_review"
APPROVED = "approved"
SKIPPED = "skipped"
POSTED = "posted"
SUPERSEDED = "superseded"  # from an interrupted run of a job that was re-run

# Post statuses (one row per clip per platform).
POST_SCHEDULED = "scheduled"
POST_POSTING = "posting"
POST_POSTED = "posted"
POST_FAILED = "failed"
POST_CANCELLED = "cancelled"


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA)
            # Columns added after the first release (CREATE TABLE IF NOT EXISTS won't add them).
            cols = {r[1] for r in con.execute("PRAGMA table_info(clips)")}
            if "vision_note" not in cols:
                con.execute("ALTER TABLE clips ADD COLUMN vision_note TEXT NOT NULL DEFAULT ''")
            if "deliver_attempts" not in cols:
                con.execute("ALTER TABLE clips ADD COLUMN deliver_attempts INTEGER NOT NULL DEFAULT 0")
            if "decided_by" not in cols:
                con.execute("ALTER TABLE clips ADD COLUMN decided_by TEXT NOT NULL DEFAULT ''")
            for col, decl in (("overlay_text", "TEXT NOT NULL DEFAULT ''"), ("caption", "TEXT NOT NULL DEFAULT ''"),
                              ("cover_offset", "REAL"), ("rank", "INTEGER")):
                if col not in cols:
                    con.execute(f"ALTER TABLE clips ADD COLUMN {col} {decl}")
            job_cols = {r[1] for r in con.execute("PRAGMA table_info(jobs)")}
            if "attempts" not in job_cols:
                con.execute("ALTER TABLE jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
            if "user_id" not in job_cols:
                con.execute("ALTER TABLE jobs ADD COLUMN user_id INTEGER")
            if "user_name" not in job_cols:
                con.execute("ALTER TABLE jobs ADD COLUMN user_name TEXT NOT NULL DEFAULT ''")
            for col, decl in (("range_start", "REAL"), ("range_end", "REAL"), ("part_label", "TEXT NOT NULL DEFAULT ''")):
                if col not in job_cols:
                    con.execute(f"ALTER TABLE jobs ADD COLUMN {col} {decl}")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=30000")
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ---- jobs ---------------------------------------------------------------

    def add_job(self, url: str, chat_id: int | None = None, status: str = QUEUED,
                user_id: int | None = None, user_name: str = "",
                range_start: float | None = None, range_end: float | None = None, part_label: str = "") -> int:
        now = time.time()
        with self.connect() as con:
            cur = con.execute(
                "INSERT INTO jobs (url, chat_id, status, user_id, user_name, range_start, range_end, part_label, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (url, chat_id, status, user_id, user_name, range_start, range_end, part_label, now, now),
            )
            return int(cur.lastrowid)

    def claim_next_job(self) -> sqlite3.Row | None:
        """Atomically move the oldest queued job to 'downloading' and return it."""
        with self.connect() as con:
            return con.execute(
                """
                UPDATE jobs SET status = 'downloading', attempts = attempts + 1, updated_at = ?
                WHERE id = (SELECT id FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1)
                RETURNING *
                """,
                (time.time(),),
            ).fetchone()

    def requeue_stuck_jobs(self) -> int:
        """After a worker crash/restart: re-queue half-finished jobs (their unreviewed clips from the
        interrupted run are superseded), fail jobs that already died MAX_ATTEMPTS times, and release
        jobs the bot never finished announcing."""
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        now = time.time()
        with self.connect() as con:
            stuck = [r["id"] for r in con.execute(
                f"SELECT id FROM jobs WHERE status IN ({placeholders})", ACTIVE_STATUSES)]
            if stuck:
                qs = ",".join("?" for _ in stuck)
                con.execute(
                    f"UPDATE clips SET status = 'superseded' WHERE status = 'pending_review' AND job_id IN ({qs})",
                    stuck,
                )
            con.execute(
                f"UPDATE jobs SET status = 'failed', updated_at = ?, error = "
                f"'The worker stopped twice while processing this job (out of memory? restart?). Send the link again to retry.' "
                f"WHERE status IN ({placeholders}) AND attempts >= ?",
                (now, *ACTIVE_STATUSES, MAX_ATTEMPTS),
            )
            cur = con.execute(
                f"UPDATE jobs SET status = 'queued', updated_at = ? WHERE status IN ({placeholders})",
                (now, *ACTIVE_STATUSES),
            )
            con.execute(
                "UPDATE jobs SET status = 'queued', updated_at = ? WHERE status = 'new' AND created_at < ?",
                (now, now - 300),
            )
            return cur.rowcount

    def update_job(self, job_id: int, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.connect() as con:
            con.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))

    def get_job(self, job_id: int) -> sqlite3.Row | None:
        with self.connect() as con:
            return con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    def recent_jobs(self, limit: int = 10) -> list[sqlite3.Row]:
        with self.connect() as con:
            return con.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    def queue_position(self, job_id: int) -> int:
        """1 = next up. Counts this job even while it is still 'new'."""
        with self.connect() as con:
            row = con.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'new') AND id <= ?", (job_id,)
            ).fetchone()
            return int(row[0])

    # ---- clips --------------------------------------------------------------

    def add_clip(self, **fields: Any) -> int:
        fields.setdefault("created_at", time.time())
        cols = ", ".join(fields)
        qs = ", ".join("?" for _ in fields)
        with self.connect() as con:
            cur = con.execute(f"INSERT INTO clips ({cols}) VALUES ({qs})", tuple(fields.values()))
            return int(cur.lastrowid)

    def update_clip(self, clip_id: int, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.connect() as con:
            con.execute(f"UPDATE clips SET {cols} WHERE id = ?", (*fields.values(), clip_id))

    def get_clip(self, clip_id: int) -> sqlite3.Row | None:
        with self.connect() as con:
            return con.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()

    def undelivered_clips(self, limit: int = 5, max_attempts: int = 3) -> list[sqlite3.Row]:
        """Clips waiting for review that never reached Telegram (oldest first). Jobs started from
        the CLI have no chat and are never pushed to Telegram."""
        with self.connect() as con:
            return con.execute(
                "SELECT clips.*, jobs.chat_id AS chat_id FROM clips JOIN jobs ON jobs.id = clips.job_id "
                "WHERE clips.status = 'pending_review' AND clips.telegram_message_id IS NULL "
                "AND clips.deliver_attempts < ? AND jobs.chat_id IS NOT NULL "
                "ORDER BY clips.id LIMIT ?",
                (max_attempts, limit),
            ).fetchall()

    def set_clip_decision(self, clip_id: int, status: str) -> None:
        self.update_clip(clip_id, status=status, decided_at=time.time())

    def recent_decisions(self, status: str, limit: int = 5) -> list[sqlite3.Row]:
        """Recently approved/skipped clips, used as taste examples for the judge."""
        with self.connect() as con:
            return con.execute(
                "SELECT * FROM clips WHERE status = ? AND decided_at IS NOT NULL "
                "ORDER BY decided_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()

    def clip_counts(self) -> dict[str, int]:
        with self.connect() as con:
            rows = con.execute("SELECT status, COUNT(*) FROM clips GROUP BY status").fetchall()
            return {r[0]: int(r[1]) for r in rows}

    # ---- posts ---------------------------------------------------------------------

    def schedule_post(self, clip_id: int, platform: str, at: float) -> None:
        """(Re)schedule a clip for a platform. A post that already went out is left alone."""
        now = time.time()
        with self.connect() as con:
            con.execute(
                "INSERT INTO posts (clip_id, platform, scheduled_at, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(clip_id, platform) DO UPDATE SET status = 'scheduled', scheduled_at = excluded.scheduled_at, "
                "attempts = 0, next_try_at = NULL, error = NULL WHERE posts.status IN ('cancelled', 'failed')",
                (clip_id, platform, at, now),
            )

    def cancel_posts(self, clip_id: int) -> int:
        with self.connect() as con:
            return con.execute(
                "UPDATE posts SET status = 'cancelled' WHERE clip_id = ? AND status = 'scheduled'", (clip_id,)
            ).rowcount

    def taken_slots(self, platform: str, since: float) -> list[float]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT scheduled_at FROM posts WHERE platform = ? AND scheduled_at >= ? "
                "AND status IN ('scheduled', 'posting', 'posted')",
                (platform, since),
            ).fetchall()
            return [float(r[0]) for r in rows]

    def due_posts(self, now: float, limit: int = 5) -> list[sqlite3.Row]:
        with self.connect() as con:
            return con.execute(
                "SELECT * FROM posts WHERE status = 'scheduled' AND scheduled_at <= ? "
                "AND (next_try_at IS NULL OR next_try_at <= ?) ORDER BY scheduled_at LIMIT ?",
                (now, now, limit),
            ).fetchall()

    def claim_post(self, post_id: int) -> bool:
        with self.connect() as con:
            return con.execute(
                "UPDATE posts SET status = 'posting' WHERE id = ? AND status = 'scheduled'", (post_id,)
            ).rowcount == 1

    def update_post(self, post_id: int, **fields: Any) -> None:
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.connect() as con:
            con.execute(f"UPDATE posts SET {cols} WHERE id = ?", (*fields.values(), post_id))

    def get_post(self, post_id: int) -> sqlite3.Row | None:
        with self.connect() as con:
            return con.execute("SELECT * FROM posts WHERE id = ?", (post_id,)).fetchone()

    def posts_for_clip(self, clip_id: int) -> list[sqlite3.Row]:
        with self.connect() as con:
            return con.execute("SELECT * FROM posts WHERE clip_id = ? ORDER BY platform", (clip_id,)).fetchall()

    def upcoming_posts(self, limit: int = 20) -> list[sqlite3.Row]:
        with self.connect() as con:
            return con.execute(
                "SELECT posts.*, clips.title AS title FROM posts JOIN clips ON clips.id = posts.clip_id "
                "WHERE posts.status IN ('scheduled', 'posting') ORDER BY posts.scheduled_at LIMIT ?",
                (limit,),
            ).fetchall()

    def requeue_stuck_posts(self) -> int:
        """A post caught mid-upload by a restart goes back to the queue (Instagram publishes only
        at the very last call, so a retry rarely duplicates)."""
        with self.connect() as con:
            return con.execute("UPDATE posts SET status = 'scheduled' WHERE status = 'posting'").rowcount

    # ---- llm usage ------------------------------------------------------------

    def record_llm_usage(
        self, provider: str, model: str, input_tokens: int, output_tokens: int, cost_usd: float
    ) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO llm_usage (ts, provider, model, input_tokens, output_tokens, cost_usd) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), provider, model, input_tokens, output_tokens, cost_usd),
            )

    def spend_this_month(self, provider: str) -> float:
        start = time.mktime(time.strptime(time.strftime("%Y-%m-01"), "%Y-%m-%d"))
        with self.connect() as con:
            row = con.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_usage WHERE provider = ? AND ts >= ?",
                (provider, start),
            ).fetchone()
            return float(row[0])

    # ---- key/value (bot conversation state) -----------------------------------

    def kv_get(self, key: str) -> str | None:
        with self.connect() as con:
            row = con.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
            return row[0] if row else None

    def kv_set(self, key: str, value: str | None) -> None:
        with self.connect() as con:
            if value is None:
                con.execute("DELETE FROM kv WHERE key = ?", (key,))
            else:
                con.execute(
                    "INSERT INTO kv (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, value),
                )
