"""Tool registry.

Tools are registered with a name, description, pydantic input/output
schemas, the specialist agents allowed to call them, and a per-minute rate
limit. Every invocation is traced (inputs, outputs, latency, success) via
orchestration.tracing.tracer, satisfying the spec's "each tool invocation is
logged" requirement without specialists having to remember to log anything
themselves.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from orchestration.tracing.tracer import traced


class ToolError(Exception):
    pass


class RateLimitExceeded(ToolError):
    pass


@dataclass
class Tool:
    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    allowed_agents: list[str]
    rate_limit_per_minute: int
    func: Callable[[BaseModel], BaseModel]

    def to_llm_schema(self) -> dict:
        """OpenAI function-calling tool schema, so agents can offer this
        tool to the LLM directly."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema.model_json_schema(),
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._call_times: dict[str, deque] = defaultdict(deque)
        # Guards _call_times: the registry is a module-level singleton, and the check-then-append
        # rate-limit sequence below isn't atomic. Without this, concurrent callers (e.g. two Streamlit
        # sessions in different threads hitting the same tool at once) could both read "under limit"
        # before either records their call, letting the limit be exceeded.
        self._rate_limit_lock = threading.Lock()

    def register(
        self,
        *,
        name: str,
        description: str,
        input_schema: type[BaseModel],
        output_schema: type[BaseModel],
        allowed_agents: list[str],
        rate_limit_per_minute: int,
        func: Callable[[BaseModel], BaseModel],
    ) -> None:
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered")
        self._tools[name] = Tool(
            name=name, description=description, input_schema=input_schema,
            output_schema=output_schema, allowed_agents=allowed_agents,
            rate_limit_per_minute=rate_limit_per_minute, func=func,
        )

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"Unknown tool '{name}'")
        return self._tools[name]

    def for_agent(self, agent: str) -> list[Tool]:
        return [t for t in self._tools.values() if agent in t.allowed_agents]

    def _check_rate_limit(self, tool: Tool) -> None:
        with self._rate_limit_lock:
            now = time.monotonic()
            window = self._call_times[tool.name]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= tool.rate_limit_per_minute:
                raise RateLimitExceeded(
                    f"Tool '{tool.name}' rate limit exceeded ({tool.rate_limit_per_minute}/min)"
                )
            window.append(now)

    def invoke(
        self, *, agent: str, task_id: str, tool_name: str, raw_input: dict, parent_span_id: str | None = None
    ) -> dict:
        tool = self.get(tool_name)
        if agent not in tool.allowed_agents:
            raise ToolError(f"Agent '{agent}' is not permitted to call tool '{tool_name}'")

        with traced(
            task_id=task_id, agent=agent, span_type="tool_call", name=tool_name,
            parent_span_id=parent_span_id, input_data=raw_input,
        ) as span:
            self._check_rate_limit(tool)
            validated_input = tool.input_schema.model_validate(raw_input)
            result: BaseModel = tool.func(validated_input)
            output = result.model_dump()
            span.result(output_data=output)
            return output


_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        _register_builtin_tools(_registry)
    return _registry


def _register_builtin_tools(registry: ToolRegistry) -> None:
    from orchestration.tools import calculator, code_exec, file_io, web_search

    web_search.register(registry)
    file_io.register(registry)
    code_exec.register(registry)
    calculator.register(registry)
