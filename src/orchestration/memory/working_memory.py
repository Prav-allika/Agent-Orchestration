"""Short-term working memory: shared state for a single task's execution.

Backed by SQLite (see db/connection.py's docstring for why this stands in
for Redis in the MVP). Scoped to one task_id and cleared when the task
completes -- nothing here is meant to outlive a single run.
"""
from __future__ import annotations

import json
from typing import Any

from orchestration.db.connection import get_connection


def set_value(task_id: str, key: str, value: Any) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO working_memory (task_id, key, value_json, updated_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(task_id, key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at
        """,
        (task_id, key, json.dumps(value, default=str)),
    )
    conn.commit()


def get_value(task_id: str, key: str, default: Any = None) -> Any:
    conn = get_connection()
    row = conn.execute(
        "SELECT value_json FROM working_memory WHERE task_id = ? AND key = ?", (task_id, key)
    ).fetchone()
    return json.loads(row["value_json"]) if row else default


def get_all(task_id: str) -> dict[str, Any]:
    conn = get_connection()
    rows = conn.execute("SELECT key, value_json FROM working_memory WHERE task_id = ?", (task_id,)).fetchall()
    return {r["key"]: json.loads(r["value_json"]) for r in rows}


def append_log(task_id: str, key: str, entry: Any, max_entries: int = 200) -> None:
    """Append to a list-valued key, used for intermediate results / error logs."""
    log = get_value(task_id, key, default=[])
    log.append(entry)
    set_value(task_id, key, log[-max_entries:])


def clear(task_id: str) -> None:
    conn = get_connection()
    conn.execute("DELETE FROM working_memory WHERE task_id = ?", (task_id,))
    conn.commit()
