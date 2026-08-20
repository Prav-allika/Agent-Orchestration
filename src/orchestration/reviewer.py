"""Reviewer agent: validates specialist output before it reaches the
supervisor/synthesis step, per the spec's three-layer hierarchy (supervisor
-> specialists -> reviewer).
"""
from __future__ import annotations

from orchestration import config
from orchestration.llm.provider import get_provider
from orchestration.schemas import ReviewResult, Subtask
from orchestration.tracing.tracer import traced

SYSTEM_PROMPT = """You are the Reviewer agent. You receive a subtask's description, its expected \
output format, and what a specialist actually produced. Judge critically:

- Does the output actually satisfy the subtask description and required inputs?
- Does it match the expected output format?
- Are there unsupported claims, obvious errors, or missing pieces?

Set passed=true only if the output is genuinely usable as-is. Give a quality_score in [0,1] \
reflecting overall quality, not just pass/fail. List concrete issues (empty list if none) so a \
specialist could act on your feedback without re-reading the whole output."""


def review_subtask_output(*, task_id: str, subtask: Subtask, output: str, parent_span_id: str | None = None) -> ReviewResult:
    provider = get_provider()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Subtask: {subtask.description}\n"
                f"Expected output format: {subtask.expected_output_format}\n\n"
                f"Specialist output:\n{output}"
            ),
        },
    ]

    with traced(
        task_id=task_id, agent="reviewer", span_type="review", name=f"review_{subtask.id}",
        parent_span_id=parent_span_id, input_data={"subtask_id": subtask.id},
    ) as span:
        review, resp = provider.complete_structured(model=config.settings.reviewer_model, messages=messages, schema=ReviewResult)
        span.result(
            output_data=review.model_dump(),
            tokens_prompt=resp.prompt_tokens, tokens_completion=resp.completion_tokens, cost_usd=resp.cost_usd,
        )
    return review
