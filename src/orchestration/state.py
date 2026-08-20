"""LangGraph state shape. A TypedDict rather than a pydantic model because
LangGraph merges partial dict returns from each node into this shape --
nodes only need to know about the state's fields.

Decision flags (needs_plan_approval, last_subtask_status, review_status,
...) are written by nodes and read by conditional-edge functions, keeping
all I/O inside nodes and all routing logic pure/side-effect-free, per
LangGraph's recommended pattern.
"""
from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    task_id: str
    user_id: str
    request: str
    explicit_review_requested: bool

    memories: list[dict]
    plan: dict[str, Any]
    needs_plan_approval: bool
    plan_status: str  # approved/rejected/took_over

    pending_approval_id: str
    pending_approval_payload: dict[str, Any]

    subtask_outputs: dict[str, str]
    subtask_retry_counts: dict[str, int]
    review_retry_counts: dict[str, int]
    approved_sensitive_subtasks: list[str]

    current_subtask_id: str | None
    subtask_needs_approval: bool
    plan_blocked: bool
    blocked_subtask_ids: list[str]
    last_subtask_status: str  # success/failed
    last_error: str | None

    last_review: dict[str, Any] | None
    review_status: str  # passed/needs_retry/needs_escalation

    resolved_action: str  # approved/modified/rejected/took_over

    final_output: str | None
    status: str  # planning/running/awaiting_approval/completed/failed
