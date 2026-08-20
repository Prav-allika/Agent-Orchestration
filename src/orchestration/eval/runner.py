"""Eval harness: runs every golden task through the real graph, scores the
result with the LLM judge, and persists everything to eval_results so runs
are comparable over time.

There's no human in an unattended eval run, so escalations need an
automated stand-in decision -- _eval_resolver plays that role. It isn't a
blanket auto-approve: a plan-level escalation triggered by low confidence
alone means the supervisor doesn't have enough grounding to act, and a
competent reviewer would decline to let it guess rather than rubber-stamp
"proceed anyway." Sensitivity-triggered and action-level escalations are
still approved, since those aren't signals the plan itself is unsound.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from orchestration.db.connection import get_connection
from orchestration.eval.golden_tasks import GOLDEN_TASKS, GoldenTask
from orchestration.eval.judge import judge_output
from orchestration.graph_runner import new_task_id, run_to_completion
from orchestration.tracing.tracer import get_task_cost_summary, new_id

LOW_CONFIDENCE_REJECT_THRESHOLD = 0.5


def _eval_resolver(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("kind") == "plan" and payload.get("level") == "approve_plan":
        plan = payload.get("plan", {})
        if plan.get("confidence", 1.0) < LOW_CONFIDENCE_REJECT_THRESHOLD and not plan.get("is_sensitive", False):
            return {"decision": "reject", "notes": "eval harness: insufficient grounding, not proceeding"}
    return {"decision": "approve"}


@dataclass
class EvalResult:
    golden_task_id: str
    task_id: str
    task_status: str
    final_output: str | None
    judge_passed: bool | None
    judge_score: float | None


def _run_one(eval_run_id: str, golden: GoldenTask) -> EvalResult:
    try:
        result = run_to_completion(user_id=golden.user_id, request=golden.request, resolve_escalation=_eval_resolver)
    except Exception as exc:  # noqa: BLE001 - one crashing golden task (a transient API error that
        # survives all provider retries, hitting max_escalations, etc.) must not lose every other
        # task's results -- run_eval_suite() used to be a plain list comprehension over _run_one(),
        # so an uncaught exception here aborted the whole batch. A synthetic failed task row is
        # inserted so eval_results' FK constraint is satisfiable even though the graph never got far
        # enough to create one itself.
        task_id = new_task_id()
        conn = get_connection()
        conn.execute(
            "INSERT OR IGNORE INTO tasks (id, user_id, request, status, result) VALUES (?, ?, ?, 'failed', ?)",
            (task_id, golden.user_id, golden.request, f"eval harness error: {exc}"),
        )
        conn.execute(
            """
            INSERT INTO eval_results
                (id, eval_run_id, golden_task_id, task_id, request, final_output, task_status,
                 judge_passed, judge_score, judge_json, cost_usd, latency_ms, escalation_count)
            VALUES (?, ?, ?, ?, ?, NULL, 'error', NULL, NULL, ?, 0, 0, 0)
            """,
            (new_id("evalres"), eval_run_id, golden.id, task_id, golden.request,
             json.dumps({"error": f"run_to_completion raised: {exc}"})),
        )
        conn.commit()
        return EvalResult(golden_task_id=golden.id, task_id=task_id, task_status="error",
                           final_output=None, judge_passed=None, judge_score=None)

    judge_passed: bool | None = None
    judge_score: float | None = None
    judge_json: str | None = None
    try:
        judgment = judge_output(
            task_id=result.task_id, request=golden.request, rubric=golden.rubric, final_output=result.final_output,
        )
        judge_passed, judge_score = judgment.overall_passed, judgment.overall_score
        judge_json = judgment.model_dump_json()
    except Exception as exc:  # noqa: BLE001 - a broken judge call shouldn't lose the run's other results
        judge_json = f'{{"error": "judging failed: {exc}"}}'

    cost_summary = get_task_cost_summary(result.task_id)

    conn = get_connection()
    conn.execute(
        """
        INSERT INTO eval_results
            (id, eval_run_id, golden_task_id, task_id, request, final_output, task_status,
             judge_passed, judge_score, judge_json, cost_usd, latency_ms, escalation_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            new_id("evalres"), eval_run_id, golden.id, result.task_id, golden.request, result.final_output,
            result.status, int(judge_passed) if judge_passed is not None else None, judge_score, judge_json,
            cost_summary.get("cost_usd") or 0, cost_summary.get("latency_ms_total") or 0,
            cost_summary.get("escalations") or 0,
        ),
    )
    conn.commit()

    return EvalResult(
        golden_task_id=golden.id, task_id=result.task_id, task_status=result.status,
        final_output=result.final_output, judge_passed=judge_passed, judge_score=judge_score,
    )


def run_eval_suite(tasks: list[GoldenTask] | None = None) -> dict:
    """Run the full golden dataset (or a subset) and return a summary.
    Every individual result is also persisted to eval_results regardless of
    what this function returns, so historical runs stay queryable.
    """
    eval_run_id = f"evalrun_{uuid.uuid4().hex[:10]}"
    results = [_run_one(eval_run_id, golden) for golden in (tasks or GOLDEN_TASKS)]

    passed = sum(1 for r in results if r.judge_passed)
    judged = sum(1 for r in results if r.judge_passed is not None)
    return {
        "eval_run_id": eval_run_id,
        "results": results,
        "pass_rate": passed / judged if judged else None,
        "passed": passed,
        "judged": judged,
        "total": len(results),
    }


def get_eval_run_history() -> list[dict]:
    """Rollup of every past eval_run_id, most recent first -- for tracking
    whether golden-task pass rate is improving or regressing over time.
    """
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT eval_run_id,
               MIN(created_at) AS started_at,
               COUNT(*) AS total,
               SUM(judge_passed) AS passed,
               SUM(cost_usd) AS cost_usd
        FROM eval_results GROUP BY eval_run_id ORDER BY started_at DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def get_eval_run_detail(eval_run_id: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM eval_results WHERE eval_run_id = ? ORDER BY created_at ASC", (eval_run_id,)
    ).fetchall()
    return [dict(r) for r in rows]
