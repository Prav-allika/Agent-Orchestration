"""Execution tracing: every agent decision, tool call, and review becomes a
trace_spans row. This is deliberately a thin SQLite table rather than a full
OpenTelemetry collector -- the spec calls for OTel spans, but for a
single-process MVP a table gives the same "tree of spans with attributes"
shape the trace explorer UI needs, without standing up a collector.
Swapping in real OTel later means changing this module only.
"""
from __future__ import annotations

import contextlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from orchestration.db.connection import get_connection


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Span:
    id: str
    task_id: str
    agent: str
    span_type: str
    name: str
    parent_span_id: str | None = None
    _start: float = field(default_factory=time.perf_counter, repr=False)

    def finish(
        self,
        *,
        status: str = "success",
        input_data: Any = None,
        output_data: Any = None,
        error: str | None = None,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        latency_ms = (time.perf_counter() - self._start) * 1000
        conn = get_connection()
        conn.execute(
            """
            UPDATE trace_spans
            SET status = ?, output_json = ?, error = ?, latency_ms = ?,
                tokens_prompt = tokens_prompt + ?, tokens_completion = tokens_completion + ?,
                cost_usd = cost_usd + ?, ended_at = datetime('now')
            WHERE id = ?
            """,
            (
                status,
                json.dumps(output_data, default=str) if output_data is not None else None,
                error,
                latency_ms,
                tokens_prompt,
                tokens_completion,
                cost_usd,
                self.id,
            ),
        )
        conn.commit()


def start_span(
    *,
    task_id: str,
    agent: str,
    span_type: str,
    name: str,
    parent_span_id: str | None = None,
    input_data: Any = None,
) -> Span:
    span_id = new_id("span")
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO trace_spans (id, task_id, parent_span_id, agent, span_type, name, input_json, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
        """,
        (span_id, task_id, parent_span_id, agent, span_type, name,
         json.dumps(input_data, default=str) if input_data is not None else None),
    )
    conn.commit()
    return Span(id=span_id, task_id=task_id, agent=agent, span_type=span_type, name=name, parent_span_id=parent_span_id)


@contextlib.contextmanager
def traced(
    *,
    task_id: str,
    agent: str,
    span_type: str,
    name: str,
    parent_span_id: str | None = None,
    input_data: Any = None,
):
    """Context manager wrapping a unit of work in a span.

    Usage::

        with traced(task_id=t, agent="research", span_type="tool_call", name="web_search") as span:
            result = do_work()
            span.result(output_data=result)

    On an uncaught exception the span is finished with status=failure and
    the exception re-raised.
    """
    span = start_span(
        task_id=task_id, agent=agent, span_type=span_type, name=name,
        parent_span_id=parent_span_id, input_data=input_data,
    )
    holder: dict[str, Any] = {"output": None, "tokens_prompt": 0, "tokens_completion": 0, "cost_usd": 0.0}

    def result(*, output_data: Any = None, tokens_prompt: int = 0, tokens_completion: int = 0, cost_usd: float = 0.0):
        holder["output"] = output_data
        holder["tokens_prompt"] = tokens_prompt
        holder["tokens_completion"] = tokens_completion
        holder["cost_usd"] = cost_usd

    span.result = result  # type: ignore[attr-defined]
    try:
        yield span
    except Exception as exc:
        span.finish(status="failure", error=str(exc))
        raise
    else:
        span.finish(
            status="success",
            output_data=holder["output"],
            tokens_prompt=holder["tokens_prompt"],
            tokens_completion=holder["tokens_completion"],
            cost_usd=holder["cost_usd"],
        )


def get_trace_tree(task_id: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM trace_spans WHERE task_id = ? ORDER BY started_at ASC", (task_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_task_cost_summary(task_id: str) -> dict:
    conn = get_connection()
    row = conn.execute(
        """
        SELECT COUNT(*) AS span_count,
               SUM(tokens_prompt) AS tokens_prompt,
               SUM(tokens_completion) AS tokens_completion,
               SUM(cost_usd) AS cost_usd,
               SUM(latency_ms) AS latency_ms_total,
               SUM(CASE WHEN status = 'failure' THEN 1 ELSE 0 END) AS failures,
               SUM(CASE WHEN status = 'escalated' THEN 1 ELSE 0 END) AS escalations
        FROM trace_spans WHERE task_id = ?
        """,
        (task_id,),
    ).fetchone()
    return dict(row) if row else {}


def get_aggregate_stats() -> dict:
    """Rollup across all tasks: cost per agent, tool usage, escalation rate."""
    conn = get_connection()
    by_agent = conn.execute(
        """
        SELECT agent, COUNT(*) AS spans, SUM(cost_usd) AS cost_usd, SUM(latency_ms) AS latency_ms,
               SUM(tokens_prompt) AS tokens_prompt, SUM(tokens_completion) AS tokens_completion
        FROM trace_spans GROUP BY agent ORDER BY cost_usd DESC
        """
    ).fetchall()
    by_tool = conn.execute(
        """
        SELECT name, COUNT(*) AS calls,
               SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) AS successes,
               AVG(latency_ms) AS avg_latency_ms
        FROM trace_spans WHERE span_type = 'tool_call' GROUP BY name ORDER BY calls DESC
        """
    ).fetchall()
    totals = conn.execute(
        """
        SELECT COUNT(DISTINCT task_id) AS total_tasks,
               SUM(cost_usd) AS total_cost_usd,
               (SELECT COUNT(*) FROM approval_requests) AS total_escalations
        FROM trace_spans
        """
    ).fetchone()
    return {
        "by_agent": [dict(r) for r in by_agent],
        "by_tool": [dict(r) for r in by_tool],
        "totals": dict(totals) if totals else {},
    }


def get_quality_metrics() -> dict:
    """Correctness/quality signals, as distinct from the cost/latency
    "did it run" metrics in get_aggregate_stats(). These answer "is the
    system actually doing the right thing", not just "did it complete":

    - task_success_rate: completed vs failed/rejected tasks
    - reviewer_first_pass_rate: fraction of subtasks the reviewer accepted
      on the FIRST attempt (a retry means the specialist got it wrong once)
    - human_override_rate: fraction of resolved escalations where the human
      did NOT simply approve the agent's proposal as-is (modified/rejected/
      took over) -- a high rate means escalation is catching real problems,
      a near-zero rate means either the system is genuinely good or the
      human reviewer isn't being critical. This number alone can't tell you
      which; read it alongside who's actually resolving the queue.
    """
    conn = get_connection()

    task_rows = conn.execute("SELECT status, COUNT(*) AS n FROM tasks GROUP BY status").fetchall()
    task_counts = {r["status"]: r["n"] for r in task_rows}
    total_tasks = sum(task_counts.values())
    completed = task_counts.get("completed", 0)

    first_pass_row = conn.execute(
        """
        WITH ranked_reviews AS (
            SELECT output_json,
                   ROW_NUMBER() OVER (PARTITION BY task_id, name ORDER BY started_at) AS attempt_num
            FROM trace_spans
            WHERE span_type = 'review' AND status = 'success'
        )
        SELECT COUNT(*) AS first_attempts,
               SUM(CASE WHEN json_extract(output_json, '$.passed') = 1 THEN 1 ELSE 0 END) AS first_attempt_passes
        FROM ranked_reviews WHERE attempt_num = 1
        """
    ).fetchone()
    first_attempts = first_pass_row["first_attempts"] or 0
    first_attempt_passes = first_pass_row["first_attempt_passes"] or 0

    resolution_rows = conn.execute(
        "SELECT status, COUNT(*) AS n FROM approval_requests WHERE status != 'pending' GROUP BY status"
    ).fetchall()
    resolution_counts = {r["status"]: r["n"] for r in resolution_rows}
    total_resolved = sum(resolution_counts.values())
    overridden = total_resolved - resolution_counts.get("approved", 0)

    return {
        "task_success_rate": {"completed": completed, "total": total_tasks,
                               "rate": completed / total_tasks if total_tasks else None},
        "reviewer_first_pass_rate": {"passed": first_attempt_passes, "total": first_attempts,
                                      "rate": first_attempt_passes / first_attempts if first_attempts else None},
        "human_override_rate": {"overridden": overridden, "total": total_resolved,
                                 "rate": overridden / total_resolved if total_resolved else None,
                                 "breakdown": resolution_counts},
    }
