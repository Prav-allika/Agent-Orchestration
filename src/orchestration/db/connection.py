"""Shared SQLite connection helper.

SQLite stands in for Postgres (system-of-record) and Redis (working memory)
in this MVP -- one file, no servers to run, same access patterns. See
.env.example for the swap-later rationale.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from orchestration import config

_local = threading.local()


def _schema_path() -> Path:
    return Path(__file__).parent / "schema.sql"


def get_connection() -> sqlite3.Connection:
    """Return a thread-local connection with the schema applied.

    The CREATE TABLE IF NOT EXISTS statements are cheap and idempotent, so
    each new thread's connection just re-applies them once against the
    shared on-disk file rather than coordinating a one-time global init.
    """
    if not hasattr(_local, "conn"):
        db_path = config.settings.resolved_sqlite_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(_schema_path().read_text())
        conn.commit()
        _local.conn = conn

    return _local.conn
