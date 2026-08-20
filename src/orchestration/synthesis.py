"""Final synthesis: combine all reviewed subtask outputs into the single
deliverable handed back to the user.
"""
from __future__ import annotations

from orchestration import config
from orchestration.llm.provider import get_provider
from orchestration.schemas import ExecutionPlan
from orchestration.tracing.tracer import traced

SYSTEM_PROMPT = """You are the Supervisor synthesizing the final deliverable from your specialists' \
reviewed outputs. Combine them into a single coherent response to the original user request. Do not \
introduce new facts -- only organize, connect, and present what the specialists produced.

Preserve evidence, don't just keep conclusions:
- If a specialist cited sources (URLs, publication names), carry every citation through to the final \
answer in the same place it supports a claim. A number or claim with no source in the input must not \
gain a bare, unsourced assertion in your output -- keep the citation attached to it.
- If a specialist's output contains a formula, expression, or arithmetic (e.g. "10000 * 1.06**5" or \
"principal * rate * time"), COPY that expression into your final answer next to the result -- do not \
paraphrase it down to a bare claim like "calculated via code" or "code was executed." A reader of your \
final answer must be able to see what was computed, not just that computation happened somewhere \
upstream. Example of what NOT to do: "the monthly payment, calculated via code, is $599.42". Example \
of what TO do: "monthly payment = 20000 * (0.05/12) / (1 - (1+0.05/12)^-36) = $599.42".
- If a specialist flagged that information was missing, unavailable, or a placeholder, keep that \
caveat in the final answer. Do not smooth it into a confident, unqualified statement -- a synthesized \
answer that reads as more certain than its inputs were is a fabrication you introduced, even if every \
individual fact traces back to a specialist.
Losing this supporting detail during synthesis makes the final answer look unsupported even when the \
underlying work was rigorous."""


def synthesize(*, task_id: str, request: str, plan: ExecutionPlan, subtask_outputs: dict[str, str]) -> str:
    provider = get_provider()
    outputs_block = "\n\n".join(
        f"[{st.id} - {st.specialist} - {st.description}]:\n{subtask_outputs.get(st.id, '(missing)')}"
        for st in plan.subtasks
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Original request:\n{request}\n\nSpecialist outputs:\n{outputs_block}"},
    ]

    with traced(task_id=task_id, agent="supervisor", span_type="synthesis", name="synthesize") as span:
        resp = provider.complete(model=config.settings.supervisor_model, messages=messages)
        span.result(output_data={"final": resp.content}, tokens_prompt=resp.prompt_tokens,
                    tokens_completion=resp.completion_tokens, cost_usd=resp.cost_usd)
    return resp.content
