"""Lets a human reviewer ask the agent clarifying questions about a pending
approval before deciding (spec Phase 3.4). A plain LLM call grounded in the
approval's packaged context and prior chat turns -- it doesn't re-invoke the
graph or specialists, just explains the decision point.
"""
from __future__ import annotations

import json

from orchestration import config
from orchestration.llm.provider import get_provider

SYSTEM_PROMPT = """You are helping a human reviewer decide whether to approve, reject, or modify a \
pending agent action. Answer their questions using ONLY the context provided below -- if something \
isn't in the context, say so rather than guessing. Be concise and direct."""


def answer_question(*, context: dict, history: list[dict], question: str) -> str:
    provider = get_provider()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context for this decision:\n{json.dumps(context, indent=2, default=str)}"},
    ]
    for turn in history:
        role = "user" if turn["role"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["content"]})
    messages.append({"role": "user", "content": question})

    resp = provider.complete(model=config.settings.specialist_model, messages=messages)
    return resp.content
