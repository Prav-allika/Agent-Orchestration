"""Full graph integration tests, with the LLM provider and embedding
function faked out so they run offline and deterministically. These prove
the wiring end-to-end: intake -> plan -> execute -> review -> synthesis ->
delivery for the happy path, and plan -> escalate -> interrupt -> resume
for the human-in-the-loop path.
"""
from orchestration.graph_runner import get_task_state, resume_task, start_task
from orchestration.hitl.approval_queue import list_pending
from orchestration.schemas import ExecutionPlan, ReviewResult, Subtask


def _simple_plan(confidence: float, sensitive: bool = False) -> ExecutionPlan:
    return ExecutionPlan(
        subtasks=[
            Subtask(id="st1", specialist="research", description="find something out", depends_on=[],
                    required_inputs="none", expected_output_format="text", complexity="low"),
        ],
        confidence=confidence, reasoning="single-step task", is_sensitive=sensitive,
    )


def test_happy_path_runs_to_completion(fake_provider, fake_embedding_function):
    fake_provider.structured_queue.append(_simple_plan(confidence=0.9))
    fake_provider.structured_queue.append(ReviewResult(passed=True, quality_score=0.9, feedback="good"))
    fake_provider.text_queue.append("This is the final synthesized answer.")

    result = start_task(user_id="u1", request="find something out")

    assert result.status == "completed"
    assert result.final_output == "This is the final synthesized answer."

    from orchestration.db.connection import get_connection

    row = get_connection().execute("SELECT status, result FROM tasks WHERE id = ?", (result.task_id,)).fetchone()
    assert row["status"] == "completed"


def test_low_confidence_plan_pauses_for_approval_then_resumes(fake_provider, fake_embedding_function):
    fake_provider.structured_queue.append(_simple_plan(confidence=0.1))

    result = start_task(user_id="u1", request="find something risky out")

    assert result.status == "awaiting_approval"
    assert result.interrupt["kind"] == "plan"
    assert "confidence" in result.interrupt["reason"]

    pending = list_pending()
    assert len(pending) == 1
    assert pending[0].task_id == result.task_id
    assert pending[0].level == "approve_plan"

    # human approves -> execution continues to completion
    fake_provider.structured_queue.append(ReviewResult(passed=True, quality_score=0.8, feedback="fine"))
    fake_provider.text_queue.append("Final answer after approval.")

    resumed = resume_task(task_id=result.task_id, resume_value={"decision": "approve"})

    assert resumed.status == "completed"
    assert resumed.final_output == "Final answer after approval."
    assert list_pending() == []


def test_rejected_plan_ends_task_as_failed_without_running_specialists(fake_provider, fake_embedding_function):
    fake_provider.structured_queue.append(_simple_plan(confidence=0.1))
    result = start_task(user_id="u1", request="do something questionable")
    assert result.status == "awaiting_approval"

    resumed = resume_task(task_id=result.task_id, resume_value={"decision": "reject", "notes": "not appropriate"})

    assert resumed.status == "failed"
    assert "not appropriate" in resumed.final_output
    # no specialist/reviewer calls should have happened after rejection
    assert not any(c["type"] == "complete_with_tools" for c in fake_provider.calls)


def test_sensitive_subtask_escalates_at_action_level(fake_provider, fake_embedding_function):
    plan = ExecutionPlan(
        subtasks=[Subtask(id="st1", specialist="writer", description="send email to the client with results",
                           depends_on=[], required_inputs="none", expected_output_format="email", complexity="low")],
        confidence=0.95, reasoning="single step", is_sensitive=False,  # planner missed it; keyword check should still catch it
    )
    fake_provider.structured_queue.append(plan)

    result = start_task(user_id="u1", request="email the client the results")

    assert result.status == "awaiting_approval"
    assert result.interrupt["kind"] == "action"
    assert "sensitive keyword" in result.interrupt["reason"]

    fake_provider.text_queue.append("Final synthesized answer.")

    resumed = resume_task(task_id=result.task_id, resume_value={"decision": "take_over", "output": "Human wrote and sent this email."})

    assert resumed.status == "completed"
    state = get_task_state(result.task_id)
    assert state["subtask_outputs"]["st1"] == "Human wrote and sent this email."


def test_rejected_action_fails_task_without_contaminating_downstream(fake_provider, fake_embedding_function):
    """Regression test: rejecting an action-level escalation used to inject
    a "[SKIPPED...]" placeholder into subtask_outputs and continue the
    graph as if that string were real specialist output -- so a downstream
    dependent subtask would receive fabricated placeholder text as its
    context, and the task would still complete. Rejection must instead end
    the task immediately, before st1 ever gets an output and before st2
    (which depends on it) ever runs.
    """
    plan = ExecutionPlan(
        subtasks=[
            Subtask(id="st1", specialist="writer", description="delete the old draft file", depends_on=[],
                    required_inputs="none", expected_output_format="confirmation", complexity="low"),
            Subtask(id="st2", specialist="writer", description="write a summary using st1's output",
                    depends_on=["st1"], required_inputs="st1 output", expected_output_format="text", complexity="low"),
        ],
        confidence=0.95, reasoning="two-step task", is_sensitive=False,
    )
    fake_provider.structured_queue.append(plan)

    result = start_task(user_id="u1", request="delete the old draft then summarize")
    assert result.status == "awaiting_approval"
    assert result.interrupt["kind"] == "action"
    assert result.interrupt["subtask"]["id"] == "st1"

    resumed = resume_task(task_id=result.task_id, resume_value={"decision": "reject", "notes": "not authorized"})

    assert resumed.status == "failed"
    assert "st1" in resumed.final_output
    assert "not authorized" in resumed.final_output

    state = get_task_state(result.task_id)
    assert "st1" not in state.get("subtask_outputs", {})  # no placeholder ever injected
    # st2 must never have been dispatched to a specialist at all
    assert not any(c["type"] == "complete_with_tools" for c in fake_provider.calls)
