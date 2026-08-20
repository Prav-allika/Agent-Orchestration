"""Entry point for starting and resuming a task's graph execution.

Both the demo CLI and the Streamlit review UI import this module rather
than touching LangGraph directly -- they run in separate OS processes, and
the only thing that lets a task started by one and approved by the other
continue correctly is that they agree on this one code path and the shared
SQLite checkpoint file.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from orchestration import config
from orchestration.graph import build_graph

_graph_definition = build_graph()


@dataclass
class TaskResult:
    task_id: str
    status: str  # running/awaiting_approval/completed/failed
    final_output: str | None
    interrupt: dict[str, Any] | None


def _extract_result(task_id: str, raw_state: dict) -> TaskResult:
    interrupts = raw_state.get("__interrupt__")
    if interrupts:
        return TaskResult(task_id=task_id, status="awaiting_approval", final_output=None, interrupt=interrupts[0].value)
    return TaskResult(
        task_id=task_id, status=raw_state.get("status", "completed"),
        final_output=raw_state.get("final_output"), interrupt=None,
    )


def new_task_id() -> str:
    return f"task_{uuid.uuid4().hex[:12]}"


def start_task(*, user_id: str, request: str, task_id: str | None = None) -> TaskResult:
    task_id = task_id or new_task_id()
    config.settings.resolved_checkpoint_path().parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(config.settings.resolved_checkpoint_path())) as saver:
        compiled = _graph_definition.compile(checkpointer=saver)
        run_config = {"configurable": {"thread_id": task_id}}
        raw_state = compiled.invoke({"task_id": task_id, "user_id": user_id, "request": request}, config=run_config)
    return _extract_result(task_id, raw_state)


def resume_task(*, task_id: str, resume_value: dict[str, Any]) -> TaskResult:
    with SqliteSaver.from_conn_string(str(config.settings.resolved_checkpoint_path())) as saver:
        compiled = _graph_definition.compile(checkpointer=saver)
        run_config = {"configurable": {"thread_id": task_id}}
        raw_state = compiled.invoke(Command(resume=resume_value), config=run_config)
    return _extract_result(task_id, raw_state)


def run_to_completion(
    *,
    user_id: str,
    request: str,
    resolve_escalation: Callable[[dict[str, Any]], dict[str, Any]] = lambda payload: {"decision": "approve"},
    max_escalations: int = 20,
) -> TaskResult:
    """Start a task and keep resuming it through however many escalations it
    hits, using `resolve_escalation` to decide each one. Default behavior
    auto-approves everything -- used by the eval harness and unattended demo
    runs, where a human isn't available to click through the Streamlit
    queue. `max_escalations` is a safety cap against a pathological loop
    (e.g. a specialist that keeps failing and keeps getting re-approved).
    """
    result = start_task(user_id=user_id, request=request)
    escalations = 0
    while result.status == "awaiting_approval":
        escalations += 1
        if escalations > max_escalations:
            raise RuntimeError(f"Task {result.task_id} exceeded {max_escalations} escalations without completing")
        decision = resolve_escalation(result.interrupt)
        result = resume_task(task_id=result.task_id, resume_value=decision)
    return result


def get_task_state(task_id: str) -> dict[str, Any] | None:
    with SqliteSaver.from_conn_string(str(config.settings.resolved_checkpoint_path())) as saver:
        compiled = _graph_definition.compile(checkpointer=saver)
        run_config = {"configurable": {"thread_id": task_id}}
        snapshot = compiled.get_state(run_config)
    return dict(snapshot.values) if snapshot else None
