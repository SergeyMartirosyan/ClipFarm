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

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_clips_status ON clips(status);
"""

# Job statuses, in pipeline order.
QUEUED = "queued"
ACTIVE_STATUSES = ("downloading", "transcribing", "analyzing", "rendering")
DONE = "done"
FAILED = "failed"

# Clip statuses.
PENDING_REVIEW = "pending_review"
APPROVED = "approved"
SKIPPED = "skipped"
POSTED = "posted"


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

    def add_job(self, url: str, chat_id: int | None = None) -> int:
        now = time.time()
        with self.connect() as con:
            cur = con.execute(
                "INSERT INTO jobs (url, chat_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (url, chat_id, now, now),
            )
            return int(cur.lastrowid)

    def claim_next_job(self) -> sqlite3.Row | None:
        """Atomically move the oldest queued job to 'downloading' and return it."""
        with self.connect() as con:
            return con.execute(
                """
                UPDATE jobs SET status = 'downloading', updated_at = ?
                WHERE id = (SELECT id FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1)
                RETURNING *
                """,
                (time.time(),),
            ).fetchone()

    def requeue_stuck_jobs(self) -> int:
        """After a worker crash/restart, put half-finished jobs back in the queue."""
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        with self.connect() as con:
            cur = con.execute(
                f"UPDATE jobs SET status = 'queued', updated_at = ? WHERE status IN ({placeholders})",
                (time.time(), *ACTIVE_STATUSES),
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
        with self.connect() as con:
            row = con.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'queued' AND id <= ?", (job_id,)
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
