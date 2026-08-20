"""Generic specialist executor: every specialist (research, data_analysis,
writer, code_exec) shares the same tool-calling loop, differing only in
system prompt and which tools the registry grants them. One executor avoids
four near-identical copies of the same loop.
"""
from __future__ import annotations

from orchestration import config
from orchestration.llm.provider import get_provider
from orchestration.schemas import Subtask
from orchestration.specialists.prompts import SYSTEM_PROMPTS
from orchestration.tools.registry import ToolError, get_registry
from orchestration.tracing.tracer import traced

MAX_TOOL_ITERATIONS = 6


class SpecialistFailure(Exception):
    pass


def run_specialist(
    *, task_id: str, subtask: Subtask, context_inputs: dict[str, str], parent_span_id: str | None = None
) -> str:
    """Run one specialist against one subtask, looping on tool calls until it
    produces a final text answer or MAX_TOOL_ITERATIONS is hit.

    context_inputs: {subtask_id: output_text} for every dependency, so the
    specialist can see what upstream subtasks produced.
    """
    provider = get_provider()
    registry = get_registry()
    tools = registry.for_agent(subtask.specialist)
    tool_schemas = [t.to_llm_schema() for t in tools]

    context_block = "\n\n".join(f"[Output of {sid}]:\n{text}" for sid, text in context_inputs.items()) or "(none)"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[subtask.specialist]},
        {
            "role": "user",
            "content": (
                f"Subtask: {subtask.description}\n"
                f"Required inputs: {subtask.required_inputs}\n"
                f"Expected output format: {subtask.expected_output_format}\n\n"
                f"Inputs from dependency subtasks:\n{context_block}"
            ),
        },
    ]

    with traced(
        task_id=task_id, agent=subtask.specialist, span_type="specialist_step", name=subtask.id,
        parent_span_id=parent_span_id, input_data={"description": subtask.description},
    ) as span:
        total_prompt_tokens = total_completion_tokens = 0
        total_cost = 0.0

        for _ in range(MAX_TOOL_ITERATIONS):
            result = provider.complete_with_tools(
                model=config.settings.specialist_model, messages=messages, tools=tool_schemas,
            )
            total_prompt_tokens += result.prompt_tokens
            total_completion_tokens += result.completion_tokens
            total_cost += result.cost_usd

            if not result.tool_calls:
                span.result(
                    output_data={"answer": result.content},
                    tokens_prompt=total_prompt_tokens, tokens_completion=total_completion_tokens,
                    cost_usd=total_cost,
                )
                return result.content or ""

            messages.append({
                "role": "assistant", "content": result.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": __import__("json").dumps(tc.arguments)}}
                    for tc in result.tool_calls
                ],
            })
            for tc in result.tool_calls:
                try:
                    output = registry.invoke(
                        agent=subtask.specialist, task_id=task_id, tool_name=tc.name,
                        raw_input=tc.arguments, parent_span_id=span.id,
                    )
                    tool_content = str(output)
                except (ToolError, Exception) as exc:  # noqa: BLE001 - surfaced back to the model, not raised
                    tool_content = f"ERROR: {exc}"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_content})

        raise SpecialistFailure(
            f"specialist '{subtask.specialist}' did not converge within {MAX_TOOL_ITERATIONS} tool-call iterations"
        )
