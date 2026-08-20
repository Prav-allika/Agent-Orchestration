"""Unit tests for individual graph nodes in isolation -- narrower and
faster than the full-graph integration tests, used here for the
low/medium-severity fixes: graceful degradation on memory-recall failure,
working memory actually being read back on a specialist retry, and
distinguishing expected specialist failures from unexpected bugs in the
error log.
"""
import orchestration.graph as graph_module
from orchestration.db.connection import get_connection
from orchestration.graph import execute_subtask_node, intake_node, review_subtask_node
from orchestration.memory import working_memory
from orchestration.schemas import ExecutionPlan, ReviewResult, Subtask
from orchestration.specialists.base import SpecialistFailure


def _task_row(task_id="t"):
    conn = get_connection()
    conn.execute("INSERT INTO tasks (id, user_id, request) VALUES (?, 'u', 'x')", (task_id,))
    conn.commit()


def test_intake_degrades_gracefully_when_memory_recall_fails(monkeypatch):
    _task_row()

    def _boom(**kwargs):
        raise RuntimeError("chroma internal error")

    monkeypatch.setattr(graph_module.long_term_memory, "recall", _boom)

    result = intake_node({"task_id": "t", "user_id": "u", "request": "do something"})

    assert result == {"memories": []}
    from orchestration.tracing.tracer import get_trace_tree

    spans = [s for s in get_trace_tree("t") if s["name"] == "memory_recall_degraded"]
    assert len(spans) == 1
    assert spans[0]["status"] == "failure"
    assert "chroma internal error" in spans[0]["error"]


def _plan_with_one_subtask() -> dict:
    plan = ExecutionPlan(
        subtasks=[Subtask(id="st1", specialist="writer", description="x", depends_on=[],
                           required_inputs="none", expected_output_format="text", complexity="low")],
        confidence=0.9, reasoning="r", is_sensitive=False,
    )
    return plan.model_dump()


def test_specialist_retry_reads_back_prior_error_from_working_memory(monkeypatch):
    _task_row()
    working_memory.append_log("t", "errors", {"subtask": "st1", "kind": "specialist_did_not_converge", "error": "loop overflow"})

    captured = {}

    def fake_run_specialist(*, task_id, subtask, context_inputs):
        captured["context_inputs"] = context_inputs
        return "final answer"

    monkeypatch.setattr(graph_module, "run_specialist", fake_run_specialist)

    state = {
        "task_id": "t", "plan": _plan_with_one_subtask(), "current_subtask_id": "st1",
        "subtask_outputs": {}, "subtask_retry_counts": {"st1": 1}, "review_retry_counts": {},
    }
    result = execute_subtask_node(state)

    assert result["last_subtask_status"] == "success"
    assert "previous_attempt_errors" in captured["context_inputs"]
    assert "loop overflow" in captured["context_inputs"]["previous_attempt_errors"]


def test_specialist_failure_tagged_distinctly_from_unexpected_error(monkeypatch):
    _task_row()
    state = {
        "task_id": "t", "plan": _plan_with_one_subtask(), "current_subtask_id": "st1",
        "subtask_outputs": {}, "subtask_retry_counts": {}, "review_retry_counts": {},
    }

    monkeypatch.setattr(graph_module, "run_specialist", lambda **kw: (_ for _ in ()).throw(SpecialistFailure("did not converge")))
    execute_subtask_node(state)
    errors = working_memory.get_value("t", "errors", default=[])
    assert errors[-1]["kind"] == "specialist_did_not_converge"

    monkeypatch.setattr(graph_module, "run_specialist", lambda **kw: (_ for _ in ()).throw(KeyError("oops, a real bug")))
    execute_subtask_node(state)
    errors = working_memory.get_value("t", "errors", default=[])
    assert errors[-1]["kind"] == "unexpected_error"


def test_review_retry_escalates_even_when_reviewer_fields_disagree(fake_provider):
    """Regression: escalation.check_review() only looks at quality_score,
    never at `passed`. If the reviewer ever returns passed=False with a
    quality_score that stays above the threshold (the two fields aren't
    actually linked, and nothing guarantees the LLM keeps them consistent),
    gating escalation on check_review() meant this condition was never true
    -- the subtask would retry forever, since subtask_retry_counts' cap only
    trips on specialist *exceptions*, not on review rejection. The retry
    cap must fire on retry count alone, independent of the score.
    """
    _task_row()
    state = {
        "task_id": "t", "plan": _plan_with_one_subtask(), "current_subtask_id": "st1",
        "subtask_outputs": {"st1": "some output"}, "review_retry_counts": {},
    }

    # First failure: passed=False but quality_score is comfortably above threshold (0.6 default)
    fake_provider.structured_queue.append(ReviewResult(passed=False, quality_score=0.9, feedback="bad", issues=["x"]))
    result1 = review_subtask_node(state)
    assert result1["review_status"] == "needs_retry"
    assert "st1" not in result1["subtask_outputs"]

    # Second failure, same inconsistent pattern -- must escalate now (default max_specialist_retries=2)
    state["review_retry_counts"] = result1["review_retry_counts"]
    state["subtask_outputs"] = {"st1": "retry output"}  # execute_subtask_node would have re-set this
    fake_provider.structured_queue.append(ReviewResult(passed=False, quality_score=0.9, feedback="still bad", issues=["x"]))
    result2 = review_subtask_node(state)
    assert result2["review_status"] == "needs_escalation"
