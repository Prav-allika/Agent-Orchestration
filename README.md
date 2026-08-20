# Agent Orchestration System

A multi-agent orchestration platform: a **supervisor** decomposes a task, delegates
subtasks to **specialist** agents with tool access, a **reviewer** validates their
output, and the system escalates to a **human** when confidence is low, an action
is sensitive, or a specialist/reviewer keeps failing — all wired on **LangGraph**,
with persistent working + long-term memory and full execution tracing.

## Architecture

```
intake -> plan -> [raise_plan_escalation -> await_plan_decision]? -> select_subtask
                                                                          |
                                          +-------------------------------+
                                          v
                    [raise_action_escalation -> await_action_decision]?
                                          |
                                    execute_subtask -> review_subtask
                                          |                 |
                                   (retry/escalate)    (retry/escalate/next)
                                          |                 |
                                          +--------> select_subtask (loop until all subtasks done)
                                                            |
                                                        synthesis -> delivery
```

- **Supervisor** (`supervisor.py`): decomposes the request into a dependency-ordered
  `ExecutionPlan` (structured output), informed by long-term memory of similar past tasks.
- **Specialists** (`specialists/`): `research`, `data_analysis`, `writer`, `code_exec` —
  one shared tool-calling executor, differing only in system prompt and tool grants.
- **Reviewer** (`reviewer.py`): scores each specialist's output before it's accepted.
- **Human-in-the-loop** (`hitl/`): pure escalation-trigger functions + a SQLite approval
  queue + LangGraph's native `interrupt()`/`Command(resume=...)` for pause/resume.
- **Memory** (`memory/`): SQLite-backed working memory (per-task, cleared on completion)
  and ChromaDB-backed long-term semantic memory (importance scoring, decay, consolidation,
  a delete endpoint).
- **Tracing** (`tracing/`): every plan/specialist-step/tool-call/review/escalation is a
  span in a SQLite table, with cost/token/latency rollups.

See [`src/orchestration/graph.py`](src/orchestration/graph.py) for the full state machine
and inline design notes (in particular: why escalation is split into a `raise_*` node and
an `await_*` node — LangGraph re-runs a node's body from the top on resume, so any node
that both has a side effect *and* calls `interrupt()` will double that side effect unless
they're split).

## MVP scope decisions

This build intentionally substitutes lighter infrastructure for the full stack described
in the original spec, to get a real, testable system running fast:

| Spec calls for | This build uses | Why |
|---|---|---|
| PostgreSQL | SQLite | One file, zero setup, same relational model |
| Redis (working memory) | SQLite table | Same access pattern (namespaced key/value) |
| Redis + Celery (async) | Synchronous in-process execution | No broker/worker to run; swap later if you need concurrency |
| OpenAI + Anthropic | OpenAI only | Only an OpenAI key was available; `llm/provider.py` is provider-agnostic — add `AnthropicProvider` and flip `PROVIDER=anthropic` |
| React review UI | Streamlit | Much faster to build a working approval queue + trace explorer |
| OpenTelemetry | SQLite `trace_spans` table | Same "tree of spans with attributes" shape without a collector |
| Docker Compose | Local venv | Nothing to containerize yet (one process, no services) |

Everything is layered so any of these can be swapped in later without touching the
agent/graph logic: `db/connection.py`, `memory/`, `llm/provider.py`, and `tracing/tracer.py`
are the seams.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -e .
cp .env.example .env   # then add your OPENAI_API_KEY
```

Requires Python 3.10+.

## Running it

**CLI demo** (the showcase scenario from the spec — research + data analysis + writing,
with a step worded to trigger sensitive-operation escalation):

```bash
python demo/run_demo.py                          # interactive: you approve/reject escalations
python demo/run_demo.py --auto-approve            # unattended
python demo/run_demo.py --memory-followup         # + a second related task to show memory informing planning
```

**Streamlit console** (submit tasks, resolve approvals, explore traces, inspect memory):

```bash
streamlit run ui/app.py
```

Pages: **Task Console** (submit + recent tasks) → **Approval Queue** (resolve escalations,
ask the agent clarifying questions) → **Trace Explorer** (execution tree, cost/perf) →
**Memory Dashboard** (what's remembered per user, decay/consolidate/delete).

## Tests

```bash
pytest
```

46 tests, fully offline — `tests/conftest.py` fakes the LLM provider and the embedding
function, so nothing here costs money or needs a network connection. Covers: plan
decomposition validity, tool sandboxing (path traversal, rate limits, agent allowlists),
reviewer rejection, memory recall/decay/consolidation/deletion, every escalation trigger,
every graph routing branch, and two full graph runs (happy path, and pause-at-escalation
→ resume-after-approval).

## What's not built yet

Per the MVP scope above: Postgres/Redis/Celery, the Anthropic provider, OpenTelemetry
export, Docker Compose, and the trace replay-with-modified-input feature (Phase 4.4).
The architecture doesn't block adding any of these — they're additive at the seams listed
above, not a rewrite.
