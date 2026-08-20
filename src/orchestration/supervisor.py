"""Supervisor agent: task decomposition. Produces a validated ExecutionPlan
(subtasks, dependencies, confidence, sensitivity) using structured output,
informed by long-term memory of similar past tasks.
"""
from __future__ import annotations

from orchestration import config
from orchestration.llm.provider import get_provider
from orchestration.memory.long_term_memory import MemoryHit
from orchestration.schemas import ExecutionPlan
from orchestration.tracing.tracer import traced

SYSTEM_PROMPT = """You are the Supervisor agent in a multi-agent orchestration system. Given a \
user request, decompose it into an ordered list of subtasks, each assigned to exactly one \
specialist:

- research: web search and fact-finding
- data_analysis: computation, data transformation, validation
- writer: prose synthesis and structured written deliverables
- code_exec: running code to produce a concrete artifact/result

Rules:
- Keep the plan as small as it can be while still fully satisfying the request (2-5 subtasks is typical).
- Express dependencies via depends_on (subtask ids that must complete first).
- Set confidence honestly: lower it when the request is ambiguous, the domain is unfamiliar, or \
you're not sure a subtask will succeed.
- Confidence in your PLAN STRUCTURE is not the same as confidence you have the FACTS the request \
needs -- don't conflate them. Knowing what a typical quarterly report looks like doesn't mean you \
know what THIS report should say. If the request references unstated prior context you don't \
actually have ("the thing we discussed", "like we agreed", "the usual format", a person/project/doc \
not described in the request or in the retrieved memory below), that missing grounding must pull \
confidence down sharply (below 0.5), even if you're structurally confident you could execute a plan. \
Do not paper over missing facts by planning subtasks that would just invent plausible-sounding content.
- Set is_sensitive=true if ANY subtask involves financial transactions, permanent data deletion, \
or external communications (sending email, posting publicly, messaging a third party) -- even if \
you're confident about the rest of the plan.
- Use the provided memory of past similar tasks to inform your plan (reuse what worked, avoid what \
didn't), but don't force-fit a mismatched past approach onto a different request. Memory of a *similar* \
past request is not the same as grounding for *this* request's specific facts."""


def _format_memories(memories: list[MemoryHit]) -> str:
    if not memories:
        return "(no relevant memory found)"
    lines = []
    for m in memories:
        lines.append(f"- [{m.kind}, importance={m.importance:.2f}] {m.text}")
    return "\n".join(lines)


def create_plan(*, task_id: str, request: str, memories: list[MemoryHit]) -> ExecutionPlan:
    provider = get_provider()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"User request:\n{request}\n\n"
                f"Relevant memory from past tasks/preferences:\n{_format_memories(memories)}"
            ),
        },
    ]

    with traced(task_id=task_id, agent="supervisor", span_type="plan", name="create_plan", input_data={"request": request}) as span:
        plan, resp = provider.complete_structured(model=config.settings.supervisor_model, messages=messages, schema=ExecutionPlan)
        span.result(
            output_data=plan.model_dump(),
            tokens_prompt=resp.prompt_tokens, tokens_completion=resp.completion_tokens, cost_usd=resp.cost_usd,
        )
    return plan
