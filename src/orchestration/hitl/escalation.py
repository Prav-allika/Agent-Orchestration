"""Escalation triggers: pure functions deciding *whether* and at *what
level* to hand control to a human. Kept separate from approval_queue.py
(which handles *persisting and waiting on* an escalation) and from the
graph (which handles *routing* around one) so each concern can be tested
and reasoned about independently.
"""
from __future__ import annotations

from dataclasses import dataclass

from orchestration import config
from orchestration.schemas import EscalationLevel, ExecutionPlan, ReviewResult

SENSITIVE_KEYWORDS = (
    "delete", "remove permanently", "transfer money", "send payment", "purchase",
    "wire transfer", "send email", "post to", "publish", "message customer", "charge card",
)


@dataclass
class EscalationDecision:
    level: str
    reason: str


def check_plan(plan: ExecutionPlan) -> EscalationDecision | None:
    """Called right after the supervisor produces a plan, before any
    specialist runs. Low confidence -> full plan review; sensitive content
    -> full plan review even if confidence is high (spec: "sensitive
    operations... financial transactions, data deletion, external
    communications" always escalate).
    """
    if plan.is_sensitive:
        return EscalationDecision(EscalationLevel.APPROVE_PLAN, "plan involves a sensitive operation")
    if plan.confidence < config.settings.confidence_threshold:
        return EscalationDecision(
            EscalationLevel.APPROVE_PLAN,
            f"planning confidence {plan.confidence:.2f} below threshold {config.settings.confidence_threshold:.2f}",
        )
    return None


def check_subtask_text(description: str, expected_output_format: str) -> EscalationDecision | None:
    """Catches sensitive subtasks the planner didn't flag as is_sensitive
    (belt-and-suspenders keyword check), evaluated before a specialist acts
    on that subtask.
    """
    haystack = f"{description} {expected_output_format}".lower()
    for kw in SENSITIVE_KEYWORDS:
        if kw in haystack:
            return EscalationDecision(EscalationLevel.APPROVE_ACTION, f"subtask text matched sensitive keyword '{kw}'")
    return None


def check_specialist_failure(retry_count: int) -> EscalationDecision | None:
    """A specialist failing twice on the same subtask escalates rather than
    retrying indefinitely.
    """
    if retry_count >= config.settings.max_specialist_retries:
        return EscalationDecision(
            EscalationLevel.APPROVE_ACTION,
            f"specialist failed {retry_count} times on the same subtask",
        )
    return None


def check_review(review: ReviewResult) -> EscalationDecision | None:
    if review.quality_score < config.settings.review_score_threshold:
        return EscalationDecision(
            EscalationLevel.APPROVE_ACTION,
            f"reviewer quality score {review.quality_score:.2f} below threshold {config.settings.review_score_threshold:.2f}",
        )
    return None


def explicit_user_request(requested: bool) -> EscalationDecision | None:
    if requested:
        return EscalationDecision(EscalationLevel.NOTIFY, "user explicitly requested human review")
    return None
