"""SQLite persistence for posts, rolling UV baselines, and alert cooldowns."""

import math
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple


SHANGHAI_TZ = timezone(timedelta(hours=8))


def now_shanghai() -> datetime:
    """Return current timezone-aware datetime in Asia/Shanghai (UTC+8)."""
    return datetime.now(SHANGHAI_TZ)


class MonitorStorage:
    """Manages SQLite tables for posts, UV baselines, and alert history."""

    def __init__(self, db_path: str = "monitor_data.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    author_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    simhash TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(platform, symbol, author_id, published_at, simhash)
                );

                CREATE INDEX IF NOT EXISTS idx_posts_symbol_time
                    ON posts(symbol, published_at);

                CREATE TABLE IF NOT EXISTS uv_baselines (
                    symbol TEXT NOT NULL,
                    hour_slot INTEGER NOT NULL,
                    sample_count INTEGER NOT NULL DEFAULT 0,
                    ewma_mean REAL NOT NULL DEFAULT 0.0,
                    ewma_var REAL NOT NULL DEFAULT 1.0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (symbol, hour_slot)
                );

                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    summary TEXT NOT NULL,
                    alerted_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_alert_symbol_time
                    ON alert_history(symbol, alerted_at);
                """
            )

    def save_posts(self, posts: List[Dict]) -> int:
        """Insert posts deduplicating by platform/symbol/author/time/simhash."""
        if not posts:
            return 0
        now_iso = now_shanghai().isoformat()
        inserted = 0
        with self._connect() as conn:
            for p in posts:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO posts
                        (symbol, platform, author_id, content, simhash, published_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        p["symbol"],
                        p["platform"],
                        p["author_id"],
                        p["content"],
                        p["simhash"],
                        p["published_at"],
                        now_iso,
                    ),
                )
                inserted += cur.rowcount
        return inserted

    def get_recent_posts(
        self,
        symbol: str,
        hours: int = 2,
        reference_time: Optional[datetime] = None,
    ) -> List[Dict]:
        """Fetch posts for a symbol within the trailing `hours` window."""
        ref = reference_time or now_shanghai()
        cutoff = (ref - timedelta(hours=hours)).isoformat()
        ref_iso = ref.isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT symbol, platform, author_id, content, simhash, published_at
                FROM posts
                WHERE symbol = ? AND published_at >= ? AND published_at <= ?
                ORDER BY published_at ASC
                """,
                (symbol, cutoff, ref_iso),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_uv_baseline(self, symbol: str, hour_slot: int) -> Tuple[float, float, int]:
        """Return (ewma_mean, ewma_std, sample_count) for a symbol and 2h slot."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT ewma_mean, ewma_var, sample_count
                FROM uv_baselines
                WHERE symbol = ? AND hour_slot = ?
                """,
                (symbol, hour_slot),
            ).fetchone()
        if not row:
            return 0.0, 1.0, 0
        std = math.sqrt(max(row["ewma_var"], 1e-6))
        return row["ewma_mean"], std, row["sample_count"]

    def update_uv_baseline(
        self,
        symbol: str,
        hour_slot: int,
        current_uv: int,
        alpha: float = 0.25,
    ) -> None:
        """Update EWMA mean and variance for a symbol's 2h time slot."""
        mean, std, count = self.get_uv_baseline(symbol, hour_slot)
        now_iso = now_shanghai().isoformat()
        if count == 0:
            new_mean = float(current_uv)
            new_var = max(float(current_uv) * 0.25, 1.0)
        else:
            diff = current_uv - mean
            new_mean = (1 - alpha) * mean + alpha * current_uv
            new_var = (1 - alpha) * (std ** 2) + alpha * (diff ** 2)
            new_var = max(new_var, 1.0)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO uv_baselines (symbol, hour_slot, sample_count, ewma_mean, ewma_var, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, hour_slot) DO UPDATE SET
                    sample_count = excluded.sample_count,
                    ewma_mean = excluded.ewma_mean,
                    ewma_var = excluded.ewma_var,
                    updated_at = excluded.updated_at
                """,
                (symbol, hour_slot, count + 1, new_mean, new_var, now_iso),
            )

    def is_in_cooldown(
        self,
        symbol: str,
        cooldown_hours: int = 6,
        reference_time: Optional[datetime] = None,
    ) -> bool:
        """Check whether an alert was already sent for `symbol` within cooldown."""
        ref = reference_time or now_shanghai()
        cutoff = (ref - timedelta(hours=cooldown_hours)).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM alert_history
                WHERE symbol = ? AND alerted_at >= ?
                LIMIT 1
                """,
                (symbol, cutoff),
            ).fetchone()
        return row is not None

    def record_alert(
        self,
        symbol: str,
        regime: str,
        risk_score: float,
        summary: str,
        reference_time: Optional[datetime] = None,
    ) -> None:
        """Record a sent alert to enforce cooldown and audit history."""
        ts = (reference_time or now_shanghai()).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO alert_history (symbol, regime, risk_score, summary, alerted_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (symbol, regime, risk_score, summary, ts),
            )

    def prune_old_posts(self, keep_days: int = 14) -> int:
        """Delete posts older than `keep_days` to keep the SQLite file compact for Git."""
        cutoff = (now_shanghai() - timedelta(days=keep_days)).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM posts WHERE published_at < ?",
                (cutoff,),
            )
            deleted = cur.rowcount
        if deleted > 0:
            with self._connect() as conn:
                conn.execute("VACUUM")
        return deleted
