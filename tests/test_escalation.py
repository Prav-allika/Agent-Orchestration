from orchestration import config as config_module
from orchestration.hitl import escalation
from orchestration.schemas import EscalationLevel, ExecutionPlan, ReviewResult, Subtask


def _plan(confidence=0.9, sensitive=False):
    return ExecutionPlan(
        subtasks=[Subtask(id="st1", description="do a thing", specialist="research", depends_on=[],
                           required_inputs="none", expected_output_format="text", complexity="low")],
        confidence=confidence, reasoning="because", is_sensitive=sensitive,
    )


def test_low_confidence_plan_escalates():
    decision = escalation.check_plan(_plan(confidence=0.1))
    assert decision is not None
    assert decision.level == EscalationLevel.APPROVE_PLAN


def test_high_confidence_plan_does_not_escalate():
    assert escalation.check_plan(_plan(confidence=0.95)) is None


def test_sensitive_plan_escalates_even_with_high_confidence():
    decision = escalation.check_plan(_plan(confidence=0.99, sensitive=True))
    assert decision is not None
    assert decision.level == EscalationLevel.APPROVE_PLAN


def test_sensitive_keyword_in_subtask_text_escalates():
    decision = escalation.check_subtask_text("send email to the customer with an update", "email body")
    assert decision is not None
    assert decision.level == EscalationLevel.APPROVE_ACTION


def test_benign_subtask_text_does_not_escalate():
    assert escalation.check_subtask_text("summarize the article", "markdown text") is None


def test_specialist_failure_escalates_at_configured_threshold():
    assert escalation.check_specialist_failure(config_module.settings.max_specialist_retries - 1) is None
    assert escalation.check_specialist_failure(config_module.settings.max_specialist_retries) is not None


def test_low_review_score_escalates():
    bad = ReviewResult(passed=False, quality_score=0.1, feedback="too short", issues=["missing citations"])
    decision = escalation.check_review(bad)
    assert decision is not None
    assert decision.level == EscalationLevel.APPROVE_ACTION


def test_good_review_score_does_not_escalate():
    good = ReviewResult(passed=True, quality_score=0.95, feedback="looks great")
    assert escalation.check_review(good) is None


def test_explicit_user_request_maps_to_notify_level():
    decision = escalation.explicit_user_request(True)
    assert decision is not None
    assert decision.level == EscalationLevel.NOTIFY
    assert escalation.explicit_user_request(False) is None
