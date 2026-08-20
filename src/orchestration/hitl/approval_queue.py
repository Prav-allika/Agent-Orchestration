"""Approval queue: persistence for escalations, plus the chat-with-the-agent
side channel the review UI offers a human before they decide. The graph
pushes requests here and blocks (via LangGraph's interrupt); the Streamlit
review UI reads/resolves them; both talk to this module, never to the
approval_requests table directly, so the schema can change in one place.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from orchestration.db.connection import get_connection
from orchestration.tracing.tracer import new_id


@dataclass
class ApprovalRequest:
    id: str
    task_id: str
    level: str
    trigger_reason: str
    context: dict
    proposed_action: dict | None
    status: str
    resolution: dict | None
    reviewer_notes: str | None
    created_at: str
    resolved_at: str | None


def _row_to_request(row) -> ApprovalRequest:
    return ApprovalRequest(
        id=row["id"], task_id=row["task_id"], level=row["level"], trigger_reason=row["trigger_reason"],
        context=json.loads(row["context_json"]),
        proposed_action=json.loads(row["proposed_action_json"]) if row["proposed_action_json"] else None,
        status=row["status"],
        resolution=json.loads(row["resolution_json"]) if row["resolution_json"] else None,
        reviewer_notes=row["reviewer_notes"], created_at=row["created_at"], resolved_at=row["resolved_at"],
    )


def create(
    *, task_id: str, level: str, trigger_reason: str, context: dict, proposed_action: dict | None = None
) -> ApprovalRequest:
    req_id = new_id("appr")
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO approval_requests (id, task_id, level, trigger_reason, context_json, proposed_action_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (req_id, task_id, level, trigger_reason, json.dumps(context, default=str),
         json.dumps(proposed_action, default=str) if proposed_action else None),
    )
    conn.execute("UPDATE tasks SET status = 'awaiting_approval' WHERE id = ?", (task_id,))
    conn.commit()
    return get(req_id)


def get(request_id: str) -> ApprovalRequest | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM approval_requests WHERE id = ?", (request_id,)).fetchone()
    return _row_to_request(row) if row else None


def list_pending() -> list[ApprovalRequest]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM approval_requests WHERE status = 'pending' ORDER BY created_at ASC"
    ).fetchall()
    return [_row_to_request(r) for r in rows]


def list_for_task(task_id: str) -> list[ApprovalRequest]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM approval_requests WHERE task_id = ? ORDER BY created_at ASC", (task_id,)
    ).fetchall()
    return [_row_to_request(r) for r in rows]


def resolve(
    request_id: str, *, status: str, resolution: dict[str, Any] | None = None, reviewer_notes: str | None = None
) -> ApprovalRequest:
    """status: approved/rejected/modified/take_over/notified."""
    conn = get_connection()
    conn.execute(
        """
        UPDATE approval_requests
        SET status = ?, resolution_json = ?, reviewer_notes = ?, resolved_at = datetime('now')
        WHERE id = ?
        """,
        (status, json.dumps(resolution, default=str) if resolution else None, reviewer_notes, request_id),
    )
    conn.commit()
    return get(request_id)


def add_chat_message(approval_id: str, *, role: str, content: str) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO approval_chat_messages (id, approval_id, role, content) VALUES (?, ?, ?, ?)",
        (new_id("chat"), approval_id, role, content),
    )
    conn.commit()


def get_chat_messages(approval_id: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT role, content, created_at FROM approval_chat_messages WHERE approval_id = ? ORDER BY created_at ASC",
        (approval_id,),
    ).fetchall()
    return [dict(r) for r in rows]
