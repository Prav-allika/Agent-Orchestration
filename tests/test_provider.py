"""Regression tests for the LLM provider's retry policy.

Before this fix, @retry(stop_after_attempt(3), wait_exponential(...)) had
no exception-type filter, so a permanent error (bad API key, malformed
request) got retried 3 times with exponential backoff exactly as readily
as a transient one (rate limit, connection blip) -- wasted time on errors
retrying can't fix. _retryable_exceptions() is the actual content of the
fix (which errors are worth a second attempt); the timing/backoff
mechanics themselves are tenacity's own well-tested behavior and aren't
re-verified here (that would mean real sleeps in the test suite).
"""
import openai
import pytest
from pydantic import ValidationError

from orchestration.llm.provider import _llm_retry, _retryable_exceptions


def test_retryable_exceptions_include_transient_api_errors():
    retryable = _retryable_exceptions()
    assert openai.RateLimitError in retryable
    assert openai.APIConnectionError in retryable
    assert openai.APITimeoutError in retryable
    assert openai.InternalServerError in retryable
    assert ValidationError in retryable  # our own cross-field validation on structured output


def test_retryable_exceptions_exclude_permanent_api_errors():
    retryable = _retryable_exceptions()
    assert openai.BadRequestError not in retryable
    assert openai.AuthenticationError not in retryable
    assert openai.PermissionDeniedError not in retryable
    assert openai.NotFoundError not in retryable


def test_llm_retry_does_not_retry_a_non_retryable_exception():
    calls = {"n": 0}

    @_llm_retry
    def always_fails_permanently():
        calls["n"] += 1
        raise ValueError("permanent failure, not in the retryable set")

    with pytest.raises(ValueError):
        always_fails_permanently()
    assert calls["n"] == 1  # no retries attempted
