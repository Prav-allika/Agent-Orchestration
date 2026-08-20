from orchestration.db.connection import get_connection
from orchestration.reviewer import review_subtask_output
from orchestration.schemas import ExecutionPlan, ReviewResult, Subtask
from orchestration.supervisor import create_plan


def _task_row():
    conn = get_connection()
    conn.execute("INSERT INTO tasks (id, user_id, request) VALUES ('t', 'u', 'test request')")
    conn.commit()


def test_create_plan_produces_valid_dependency_ordered_plan(fake_provider):
    _task_row()
    plan = ExecutionPlan(
        subtasks=[
            Subtask(id="st1", specialist="research", description="find facts", depends_on=[],
                    required_inputs="none", expected_output_format="bullet list", complexity="low"),
            Subtask(id="st2", specialist="writer", description="write summary", depends_on=["st1"],
                    required_inputs="st1 output", expected_output_format="prose", complexity="low"),
        ],
        confidence=0.85, reasoning="straightforward two-step task", is_sensitive=False,
    )
    fake_provider.structured_queue.append(plan)

    result = create_plan(task_id="t", request="summarize some facts", memories=[])

    assert [s.id for s in result.subtasks] == ["st1", "st2"]
    assert result.subtasks[1].depends_on == ["st1"]
    assert 0 <= result.confidence <= 1


def test_create_plan_records_planning_trace_span(fake_provider):
    _task_row()
    plan = ExecutionPlan(
        subtasks=[Subtask(id="st1", specialist="research", description="x", depends_on=[],
                           required_inputs="none", expected_output_format="text", complexity="low")],
        confidence=0.5, reasoning="uncertain", is_sensitive=False,
    )
    fake_provider.structured_queue.append(plan)

    create_plan(task_id="t", request="do x", memories=[])

    from orchestration.tracing.tracer import get_trace_tree

    spans = [s for s in get_trace_tree("t") if s["span_type"] == "plan"]
    assert len(spans) == 1
    assert spans[0]["status"] == "success"


def test_reviewer_catches_deliberately_bad_output(fake_provider):
    _task_row()
    subtask = Subtask(id="st1", specialist="writer", description="write a 300-word summary", depends_on=[],
                       required_inputs="none", expected_output_format="300-word prose", complexity="low")
    fake_provider.structured_queue.append(
        ReviewResult(passed=False, quality_score=0.15, feedback="only 10 words, far short of 300",
                     issues=["too short", "no citations"])
    )

    review = review_subtask_output(task_id="t", subtask=subtask, output="too short.")

    assert review.passed is False
    assert review.quality_score < 0.5
    assert "too short" in review.issues


def test_reviewer_passes_good_output(fake_provider):
    _task_row()
    subtask = Subtask(id="st1", specialist="writer", description="write a haiku", depends_on=[],
                       required_inputs="none", expected_output_format="3 lines", complexity="low")
    fake_provider.structured_queue.append(ReviewResult(passed=True, quality_score=0.9, feedback="good haiku"))

    review = review_subtask_output(task_id="t", subtask=subtask, output="old pond / a frog jumps in / water's sound")

    assert review.passed is True
