"""Provider-agnostic LLM interface.

Only OpenAI is implemented (per the MVP scope). Anthropic is stubbed so
adding it later is a matter of implementing AnthropicProvider and flipping
PROVIDER=anthropic -- nothing upstream (agents, graph) touches the OpenAI
SDK directly, they only see LLMResponse.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from orchestration import config
from orchestration.tracing.cost import compute_cost_usd

T = TypeVar("T", bound=BaseModel)


def _retryable_exceptions() -> tuple[type[Exception], ...]:
    """Only retry errors that have a real chance of succeeding on a second
    attempt: transient API issues, and structured-output responses that
    failed our own cross-field pydantic validation (worth resampling).
    Explicitly excludes permanent errors (bad request, auth, permission,
    not found) that tenacity used to retry blindly 3x with exponential
    backoff even though no amount of retrying fixes a bad API key.
    """
    import openai

    return (
        openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
        openai.InternalServerError, ValidationError,
    )


_llm_retry = retry(
    stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(_retryable_exceptions()),
)


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[ToolCall]
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float


# temperature defaults to None (omitted from the API call entirely) rather than some fixed value
# like 0.2 -- newer models (gpt-5-mini, gpt-5-nano, o-series) reject ANY explicit temperature other
# than their default (1) and error with "Unsupported value" if you pass one, while older models
# (gpt-4o-mini, gpt-4o) accept arbitrary values. Omitting the parameter is the only setting that
# works across both families; no caller in this codebase currently needs deterministic sampling
# badly enough to hardcode a model-specific override.
class LLMProvider:
    def complete(self, *, model: str, messages: list[dict], temperature: float | None = None) -> LLMResponse:
        raise NotImplementedError

    def complete_structured(
        self, *, model: str, messages: list[dict], schema: type[T], temperature: float | None = None
    ) -> tuple[T, LLMResponse]:
        raise NotImplementedError

    def complete_with_tools(
        self, *, model: str, messages: list[dict], tools: list[dict], temperature: float | None = None
    ) -> ChatResult:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str | None = None):
        from openai import OpenAI

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self._client = OpenAI(api_key=api_key)

    @_llm_retry
    def complete(self, *, model: str, messages: list[dict], temperature: float | None = None) -> LLMResponse:
        start = time.perf_counter()
        kwargs = {"temperature": temperature} if temperature is not None else {}
        resp = self._client.chat.completions.create(model=model, messages=messages, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=compute_cost_usd(model, prompt_tokens, completion_tokens),
            latency_ms=latency_ms,
        )

    @_llm_retry
    def complete_structured(
        self, *, model: str, messages: list[dict], schema: type[T], temperature: float | None = None
    ) -> tuple[T, LLMResponse]:
        start = time.perf_counter()
        kwargs = {"temperature": temperature} if temperature is not None else {}
        resp = self._client.beta.chat.completions.parse(
            model=model, messages=messages, response_format=schema, **kwargs
        )
        latency_ms = (time.perf_counter() - start) * 1000
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            raise ValueError(f"Model returned no parseable structured output: {resp.choices[0].message.refusal}")
        llm_response = LLMResponse(
            content=resp.choices[0].message.content or "",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=compute_cost_usd(model, prompt_tokens, completion_tokens),
            latency_ms=latency_ms,
        )
        return parsed, llm_response

    @_llm_retry
    def complete_with_tools(
        self, *, model: str, messages: list[dict], tools: list[dict], temperature: float | None = None
    ) -> ChatResult:
        import json as _json

        start = time.perf_counter()
        kwargs = {"temperature": temperature} if temperature is not None else {}
        resp = self._client.chat.completions.create(model=model, messages=messages, tools=tools or None, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        message = resp.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=_json.loads(tc.function.arguments or "{}"))
            for tc in (message.tool_calls or [])
        ]
        return ChatResult(
            content=message.content,
            tool_calls=tool_calls,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=compute_cost_usd(model, prompt_tokens, completion_tokens),
            latency_ms=latency_ms,
        )


_provider_instance: LLMProvider | None = None


def get_provider() -> LLMProvider:
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance

    if config.settings.provider == "openai":
        _provider_instance = OpenAIProvider(api_key=config.settings.openai_api_key)
    elif config.settings.provider == "anthropic":
        raise NotImplementedError(
            "Anthropic provider is not implemented in this MVP. Set PROVIDER=openai."
        )
    else:
        raise ValueError(f"Unknown provider: {config.settings.provider}")

    return _provider_instance
