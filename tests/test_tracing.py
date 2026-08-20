import json

from orchestration.db.connection import get_connection
from orchestration.hitl.approval_queue import create as create_approval
from orchestration.hitl.approval_queue import resolve as resolve_approval
from orchestration.tracing.tracer import get_quality_metrics, traced


def _task(task_id="t"):
    conn = get_connection()
    conn.execute("INSERT INTO tasks (id, user_id, request, status) VALUES (?, 'u', 'test', 'completed')", (task_id,))
    conn.commit()


def _review_span(task_id, subtask_name, passed):
    with traced(task_id=task_id, agent="reviewer", span_type="review", name=f"review_{subtask_name}") as span:
        span.result(output_data={"passed": passed, "quality_score": 1.0 if passed else 0.2})


def test_quality_metrics_on_empty_db_returns_none_rates():
    metrics = get_quality_metrics()
    assert metrics["task_success_rate"]["rate"] is None
    assert metrics["reviewer_first_pass_rate"]["rate"] is None
    assert metrics["human_override_rate"]["rate"] is None


def test_task_success_rate_counts_by_status():
    _task("t1")
    conn = get_connection()
    conn.execute("INSERT INTO tasks (id, user_id, request, status) VALUES ('t2', 'u', 'x', 'failed')")
    conn.commit()

    metrics = get_quality_metrics()
    assert metrics["task_success_rate"] == {"completed": 1, "total": 2, "rate": 0.5}


def test_reviewer_first_pass_rate_only_counts_first_attempt_per_subtask():
    _task()
    # st1 passes immediately (first-pass success)
    _review_span("t", "st1", passed=True)
    # st2 fails once, then passes on retry -- only the FIRST attempt should count against the rate
    _review_span("t", "st2", passed=False)
    _review_span("t", "st2", passed=True)

    metrics = get_quality_metrics()
    first_pass = metrics["reviewer_first_pass_rate"]
    assert first_pass["total"] == 2  # two distinct subtasks, not three review events
    assert first_pass["passed"] == 1  # only st1 passed on its first attempt
    assert first_pass["rate"] == 0.5


def test_human_override_rate_excludes_pending_and_counts_non_approved_as_overrides():
    _task()
    req1 = create_approval(task_id="t", level="approve_plan", trigger_reason="x", context={})
    req2 = create_approval(task_id="t", level="approve_action", trigger_reason="y", context={})
    req3 = create_approval(task_id="t", level="approve_action", trigger_reason="z", context={})
    resolve_approval(req1.id, status="approved")
    resolve_approval(req2.id, status="modified")
    resolve_approval(req3.id, status="rejected")
    # a 4th request left pending should not count toward the resolved total
    create_approval(task_id="t", level="approve_action", trigger_reason="w", context={})

    metrics = get_quality_metrics()
    override = metrics["human_override_rate"]
    assert override["total"] == 3
    assert override["overridden"] == 2
    assert round(override["rate"], 4) == round(2 / 3, 4)
    assert override["breakdown"] == {"approved": 1, "modified": 1, "rejected": 1}
