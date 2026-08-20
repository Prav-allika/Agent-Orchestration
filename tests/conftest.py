"""Shared test fixtures.

Two things make this codebase awkward to test without help:
1. Several modules cache singletons at module scope (DB connection, tool
   registry, LLM provider instance, Chroma collection) so state leaks
   between tests unless explicitly reset.
2. long_term_memory needs an embedding function, and the real one calls
   OpenAI. `fake_embedding_function` swaps in a deterministic local
   hash-based embedding so recall() tests don't need a real API key or
   network access, while still producing meaningful nearest-neighbor
   behavior for exact-word-overlap queries.
"""
from __future__ import annotations

import hashlib

import pytest


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "checkpoints.db"))
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("WORKDIR", str(tmp_path / "workdir"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")

    from orchestration import config as config_module

    monkeypatch.setattr(config_module, "settings", config_module.Settings())

    from orchestration.db import connection as db_connection

    if hasattr(db_connection._local, "conn"):
        db_connection._local.conn.close()
        del db_connection._local.conn

    import orchestration.tools.registry as registry_module

    monkeypatch.setattr(registry_module, "_registry", None)

    import orchestration.memory.long_term_memory as ltm_module

    monkeypatch.setattr(ltm_module, "_collection", None)
    monkeypatch.setattr(ltm_module, "_client", None)

    import orchestration.llm.provider as provider_module

    monkeypatch.setattr(provider_module, "_provider_instance", None)

    yield


class _FakeEmbeddingFunction:
    """Deterministic hash-based bag-of-words embedding -- no network calls,
    but similar/overlapping text still lands close together, which is all
    the memory-recall tests need.
    """

    def name(self):
        return "fake-hash-embedding"

    def is_legacy(self):
        return False

    def __call__(self, input):
        vecs = []
        for text in input:
            vec = [0.0] * 32
            for word in text.lower().split():
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                vec[h % 32] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vecs.append([v / norm for v in vec])
        return vecs

    def embed_query(self, input):
        return self.__call__(input)


@pytest.fixture
def fake_embedding_function(monkeypatch):
    from chromadb.utils import embedding_functions

    monkeypatch.setattr(embedding_functions, "OpenAIEmbeddingFunction", lambda **kwargs: _FakeEmbeddingFunction())


class FakeProvider:
    """Drop-in LLMProvider that returns pre-programmed responses instead of
    calling OpenAI, so agent-layer tests are deterministic and free.
    """

    def __init__(self):
        self.structured_queue: list = []
        self.text_queue: list[str] = []
        self.tool_call_queue: list = []
        self.calls: list[dict] = []

    def complete(self, *, model, messages, temperature=0.2):
        from orchestration.llm.provider import LLMResponse

        self.calls.append({"type": "complete", "model": model, "messages": messages})
        content = self.text_queue.pop(0) if self.text_queue else "OK"
        return LLMResponse(content=content, model=model, prompt_tokens=10, completion_tokens=10, cost_usd=0.001, latency_ms=1.0)

    def complete_structured(self, *, model, messages, schema, temperature=0.0):
        from orchestration.llm.provider import LLMResponse

        self.calls.append({"type": "complete_structured", "model": model, "messages": messages, "schema": schema})
        parsed = self.structured_queue.pop(0)
        resp = LLMResponse(content=parsed.model_dump_json(), model=model, prompt_tokens=20, completion_tokens=20,
                            cost_usd=0.002, latency_ms=1.0)
        return parsed, resp

    def complete_with_tools(self, *, model, messages, tools, temperature=0.2):
        from orchestration.llm.provider import ChatResult

        self.calls.append({"type": "complete_with_tools", "model": model, "messages": messages})
        if self.tool_call_queue:
            return self.tool_call_queue.pop(0)
        return ChatResult(content="final answer", tool_calls=[], model=model, prompt_tokens=15,
                           completion_tokens=15, cost_usd=0.001, latency_ms=1.0)


@pytest.fixture
def fake_provider(monkeypatch):
    import orchestration.llm.provider as provider_module

    fake = FakeProvider()
    monkeypatch.setattr(provider_module, "_provider_instance", fake)
    return fake
