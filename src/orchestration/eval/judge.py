"""LLM-as-judge: scores a completed task's final output against its golden
task's rubric. Traced like any other LLM call (span_type='eval_judge') so
judge cost/latency shows up in cost accounting alongside the run it judged.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from orchestration import config
from orchestration.llm.provider import get_provider
from orchestration.tracing.tracer import traced

SYSTEM_PROMPT = """You are grading an AI agent system's output against a fixed rubric. You did not \
write the output and have no stake in it passing -- judge critically and honestly. For each \
criterion, decide pass/fail based ONLY on the actual output text, not on what you'd expect a good \
answer to contain. If the output is missing entirely or clearly failed, fail every criterion it \
would have needed to satisfy."""


class CriterionResult(BaseModel):
    criterion: str
    passed: bool
    notes: str = Field(description="One sentence: why this passed or failed, citing the output")


class JudgeResult(BaseModel):
    criteria: list[CriterionResult]
    overall_passed: bool = Field(description="True only if the output is acceptable as a whole")
    overall_score: float = Field(ge=0, le=1, description="Fraction of criteria meaningfully satisfied")


def judge_output(*, task_id: str, request: str, rubric: list[str], final_output: str | None) -> JudgeResult:
    provider = get_provider()
    rubric_block = "\n".join(f"- {c}" for c in rubric)
    output_block = final_output if final_output else "(no output -- the task did not produce a final result)"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Original request:\n{request}\n\n"
                f"Rubric criteria:\n{rubric_block}\n\n"
                f"Actual output produced:\n{output_block}"
            ),
        },
    ]

    with traced(task_id=task_id, agent="judge", span_type="eval_judge", name="judge_output") as span:
        result, resp = provider.complete_structured(model=config.settings.reviewer_model, messages=messages, schema=JudgeResult)
        span.result(
            output_data=result.model_dump(),
            tokens_prompt=resp.prompt_tokens, tokens_completion=resp.completion_tokens, cost_usd=resp.cost_usd,
        )
    return result
