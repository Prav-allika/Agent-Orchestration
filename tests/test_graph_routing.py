"""Unit tests for the graph's conditional-edge functions in isolation.

These are pure functions over the state dict (no LLM/tool calls), so they
can be tested directly with synthetic state -- much faster and more
targeted than driving the whole graph for every branch.
"""
from orchestration.graph import (
    route_after_execute,
    route_after_plan,
    route_after_plan_decision,
    route_after_review,
    route_after_select,
)


def test_route_after_plan_escalates_when_flagged():
    assert route_after_plan({"needs_plan_approval": True}) == "raise_plan_escalation"


def test_route_after_plan_proceeds_when_not_flagged():
    assert route_after_plan({"needs_plan_approval": False}) == "select_subtask"


def test_route_after_plan_decision_approved_continues():
    assert route_after_plan_decision({"plan_status": "approved"}) == "select_subtask"


def test_route_after_plan_decision_rejected_goes_to_delivery():
    assert route_after_plan_decision({"plan_status": "rejected"}) == "delivery"


def test_route_after_plan_decision_took_over_goes_to_delivery():
    assert route_after_plan_decision({"plan_status": "took_over"}) == "delivery"


def test_route_after_select_no_more_subtasks_goes_to_synthesis():
    assert route_after_select({"current_subtask_id": None}) == "synthesis"


def test_route_after_select_sensitive_subtask_escalates():
    assert route_after_select({"current_subtask_id": "st1", "subtask_needs_approval": True}) == "raise_action_escalation"


def test_route_after_select_normal_subtask_executes():
    assert route_after_select({"current_subtask_id": "st1", "subtask_needs_approval": False}) == "execute_subtask"


def test_route_after_execute_success_goes_to_review():
    assert route_after_execute({"last_subtask_status": "success"}) == "review_subtask"


def test_route_after_execute_failure_under_retry_limit_loops_back():
    state = {"last_subtask_status": "failed", "current_subtask_id": "st1", "subtask_retry_counts": {"st1": 1}}
    assert route_after_execute(state) == "select_subtask"


def test_route_after_execute_failure_over_retry_limit_escalates():
    state = {"last_subtask_status": "failed", "current_subtask_id": "st1", "subtask_retry_counts": {"st1": 2}}
    assert route_after_execute(state) == "raise_action_escalation"


def test_route_after_review_passed_moves_on():
    assert route_after_review({"review_status": "passed"}) == "select_subtask"


def test_route_after_review_needs_retry_loops_back():
    assert route_after_review({"review_status": "needs_retry"}) == "select_subtask"


def test_route_after_review_needs_escalation_escalates():
    assert route_after_review({"review_status": "needs_escalation"}) == "raise_action_escalation"
