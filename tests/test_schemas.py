"""Regression tests for ExecutionPlan's dependency-graph validator.

Without this, a hallucinated plan (a subtask depending on an id that
doesn't exist, or a dependency cycle) would pass through unnoticed and
graph.py's _next_runnable_subtask() would silently never select the
blocked subtask(s) -- the task would still reach synthesis and get marked
"completed" with the gap papered over as "(missing)" text. See graph.py's
select_subtask_node docstring for the graph-level defense-in-depth that
backs this up.
"""
import pytest
from pydantic import ValidationError

from orchestration.schemas import ExecutionPlan, Subtask


def _subtask(id_, depends_on=()):
    return Subtask(id=id_, description="x", specialist="research", depends_on=list(depends_on),
                    required_inputs="n", expected_output_format="t", complexity="low")


def test_valid_plan_passes():
    plan = ExecutionPlan(subtasks=[_subtask("st1"), _subtask("st2", ["st1"])], confidence=0.9, reasoning="r", is_sensitive=False)
    assert [s.id for s in plan.subtasks] == ["st1", "st2"]


def test_rejects_dangling_dependency():
    with pytest.raises(ValidationError, match="nonexistent"):
        ExecutionPlan(subtasks=[_subtask("st1", ["st99"])], confidence=0.9, reasoning="r", is_sensitive=False)


def test_rejects_self_dependency_cycle():
    with pytest.raises(ValidationError, match="circular"):
        ExecutionPlan(subtasks=[_subtask("st1", ["st1"])], confidence=0.9, reasoning="r", is_sensitive=False)


def test_rejects_two_subtask_cycle():
    with pytest.raises(ValidationError, match="circular"):
        ExecutionPlan(subtasks=[_subtask("a", ["b"]), _subtask("b", ["a"])], confidence=0.9, reasoning="r", is_sensitive=False)


def test_rejects_duplicate_subtask_ids():
    with pytest.raises(ValidationError, match="duplicate"):
        ExecutionPlan(subtasks=[_subtask("a"), _subtask("a")], confidence=0.9, reasoning="r", is_sensitive=False)


def test_rejects_empty_subtask_list():
    with pytest.raises(ValidationError):
        ExecutionPlan(subtasks=[], confidence=0.9, reasoning="r", is_sensitive=False)
