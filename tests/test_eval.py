from orchestration.db.connection import get_connection
from orchestration.eval.golden_tasks import GOLDEN_TASKS, GoldenTask
from orchestration.eval.judge import CriterionResult, JudgeResult, judge_output
from orchestration.eval.runner import _eval_resolver, get_eval_run_detail, get_eval_run_history, run_eval_suite
from orchestration.schemas import ExecutionPlan, ReviewResult, Subtask


def test_eval_resolver_rejects_low_confidence_ungrounded_plan():
    payload = {"kind": "plan", "level": "approve_plan", "plan": {"confidence": 0.3, "is_sensitive": False}}
    decision = _eval_resolver(payload)
    assert decision["decision"] == "reject"


def test_eval_resolver_approves_low_confidence_if_sensitive():
    # sensitivity, not confidence, is why this escalated -- the plan itself may still be sound
    payload = {"kind": "plan", "level": "approve_plan", "plan": {"confidence": 0.3, "is_sensitive": True}}
    assert _eval_resolver(payload)["decision"] == "approve"


def test_eval_resolver_approves_high_confidence_plan():
    payload = {"kind": "plan", "level": "approve_plan", "plan": {"confidence": 0.9, "is_sensitive": False}}
    assert _eval_resolver(payload)["decision"] == "approve"


def test_eval_resolver_approves_action_level_escalations_regardless_of_plan_confidence():
    payload = {"kind": "action", "level": "approve_action", "subtask": {}}
    assert _eval_resolver(payload)["decision"] == "approve"


def test_golden_tasks_have_unique_ids_and_nonempty_rubrics():
    ids = [t.id for t in GOLDEN_TASKS]
    assert len(ids) == len(set(ids))
    assert len(GOLDEN_TASKS) >= 5
    for t in GOLDEN_TASKS:
        assert t.request.strip()
        assert len(t.rubric) >= 1


def test_judge_output_returns_structured_result(fake_provider):
    conn = get_connection()
    conn.execute("INSERT INTO tasks (id, user_id, request) VALUES ('t', 'u', 'x')")
    conn.commit()

    fake_provider.structured_queue.append(
        JudgeResult(
            criteria=[CriterionResult(criterion="says hello", passed=False, notes="output was empty")],
            overall_passed=False, overall_score=0.0,
        )
    )
    result = judge_output(task_id="t", request="say hi", rubric=["says hello"], final_output=None)
    assert result.overall_passed is False
    assert result.criteria[0].passed is False


def test_run_eval_suite_persists_result_and_summarizes(fake_provider, fake_embedding_function):
    golden = GoldenTask(id="t1", request="say hi", rubric=["says hello"])

    plan = ExecutionPlan(
        subtasks=[Subtask(id="st1", specialist="writer", description="say hi", depends_on=[],
                           required_inputs="none", expected_output_format="text", complexity="low")],
        confidence=0.9, reasoning="simple greeting task", is_sensitive=False,
    )
    fake_provider.structured_queue.append(plan)  # plan_node
    fake_provider.structured_queue.append(ReviewResult(passed=True, quality_score=0.9, feedback="fine"))  # review
    fake_provider.text_queue.append("Hello!")  # synthesis
    fake_provider.structured_queue.append(
        JudgeResult(
            criteria=[CriterionResult(criterion="says hello", passed=True, notes="output says Hello!")],
            overall_passed=True, overall_score=1.0,
        )
    )  # judge

    summary = run_eval_suite(tasks=[golden])

    assert summary["total"] == 1
    assert summary["judged"] == 1
    assert summary["passed"] == 1
    assert summary["pass_rate"] == 1.0

    detail = get_eval_run_detail(summary["eval_run_id"])
    assert len(detail) == 1
    assert detail[0]["golden_task_id"] == "t1"
    assert detail[0]["judge_passed"] == 1
    assert detail[0]["task_status"] == "completed"

    history = get_eval_run_history()
    assert history[0]["eval_run_id"] == summary["eval_run_id"]
    assert history[0]["total"] == 1
    assert history[0]["passed"] == 1


def test_run_eval_suite_survives_judge_failure(fake_provider, fake_embedding_function, monkeypatch):
    """If the judge call itself errors, the task's run should still be
    persisted (with judge_passed=None) instead of losing the whole result.
    """
    golden = GoldenTask(id="t1", request="say hi", rubric=["says hello"])
    plan = ExecutionPlan(
        subtasks=[Subtask(id="st1", specialist="writer", description="say hi", depends_on=[],
                           required_inputs="none", expected_output_format="text", complexity="low")],
        confidence=0.9, reasoning="simple", is_sensitive=False,
    )
    fake_provider.structured_queue.append(plan)
    fake_provider.structured_queue.append(ReviewResult(passed=True, quality_score=0.9, feedback="fine"))
    fake_provider.text_queue.append("Hello!")
    # No JudgeResult queued -> judge_output's complete_structured call will raise IndexError (pop from empty list)

    summary = run_eval_suite(tasks=[golden])

    assert summary["total"] == 1
    assert summary["judged"] == 0
    detail = get_eval_run_detail(summary["eval_run_id"])
    assert detail[0]["judge_passed"] is None
    assert detail[0]["task_status"] == "completed"


def test_run_eval_suite_isolates_a_crashing_task_from_the_rest(fake_provider, fake_embedding_function, monkeypatch):
    """Regression: run_eval_suite() used to be a plain list comprehension
    over _run_one() with no exception handling -- one golden task whose
    run_to_completion() call raised (e.g. a transient error surviving all
    provider retries) would abort the entire batch and lose every other
    task's results, not just that one's.
    """
    import orchestration.eval.runner as runner_module

    good = GoldenTask(id="good", request="say hi", rubric=["says hello"])
    bad = GoldenTask(id="bad", request="this one will crash", rubric=["n/a"])

    original_run = runner_module.run_to_completion

    def flaky_run(*, user_id, request, resolve_escalation):
        if request == bad.request:
            raise RuntimeError("simulated failure that survives all provider retries")
        return original_run(user_id=user_id, request=request, resolve_escalation=resolve_escalation)

    monkeypatch.setattr(runner_module, "run_to_completion", flaky_run)

    plan = ExecutionPlan(
        subtasks=[Subtask(id="st1", specialist="writer", description="say hi", depends_on=[],
                           required_inputs="none", expected_output_format="text", complexity="low")],
        confidence=0.9, reasoning="simple", is_sensitive=False,
    )
    fake_provider.structured_queue.append(plan)
    fake_provider.structured_queue.append(ReviewResult(passed=True, quality_score=0.9, feedback="fine"))
    fake_provider.text_queue.append("Hello!")
    fake_provider.structured_queue.append(
        JudgeResult(criteria=[CriterionResult(criterion="says hello", passed=True, notes="ok")],
                    overall_passed=True, overall_score=1.0)
    )

    summary = run_eval_suite(tasks=[bad, good])

    assert summary["total"] == 2
    statuses = {r.golden_task_id: r.task_status for r in summary["results"]}
    assert statuses["bad"] == "error"
    assert statuses["good"] == "completed"
    assert summary["passed"] == 1  # only "good" was judged and passed; "bad" never reached judging

    detail = get_eval_run_detail(summary["eval_run_id"])
    bad_row = next(r for r in detail if r["golden_task_id"] == "bad")
    assert bad_row["judge_passed"] is None
    assert "simulated failure" in bad_row["judge_json"]
