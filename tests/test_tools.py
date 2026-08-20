import pytest

from orchestration.db.connection import get_connection
from orchestration.tools.registry import RateLimitExceeded, ToolError, get_registry


@pytest.fixture(autouse=True)
def _task_row():
    conn = get_connection()
    conn.execute("INSERT INTO tasks (id, user_id, request) VALUES ('t', 'u', 'test')")
    conn.commit()


def test_calculator_evaluates_arithmetic():
    reg = get_registry()
    out = reg.invoke(agent="data_analysis", task_id="t", tool_name="calculator", raw_input={"expression": "(3 + 4) * 2"})
    assert out["result"] == 14.0


def test_calculator_rejects_unsupported_expression():
    reg = get_registry()
    with pytest.raises(Exception):
        reg.invoke(agent="data_analysis", task_id="t", tool_name="calculator", raw_input={"expression": "__import__('os')"})


def test_file_write_then_read_roundtrip():
    reg = get_registry()
    reg.invoke(agent="writer", task_id="t", tool_name="file_write", raw_input={"path": "note.txt", "content": "hello"})
    out = reg.invoke(agent="writer", task_id="t", tool_name="file_read", raw_input={"path": "note.txt"})
    assert out["content"] == "hello"


def test_file_read_blocks_path_traversal():
    reg = get_registry()
    with pytest.raises(PermissionError):
        reg.invoke(agent="writer", task_id="t", tool_name="file_read", raw_input={"path": "../../etc/passwd"})


def test_tool_enforces_agent_allowlist():
    reg = get_registry()
    with pytest.raises(ToolError):
        reg.invoke(agent="writer", task_id="t", tool_name="web_search", raw_input={"query": "x"})


def test_tool_enforces_rate_limit():
    reg = get_registry()
    tool = reg.get("calculator")
    tool.rate_limit_per_minute = 2
    reg.invoke(agent="data_analysis", task_id="t", tool_name="calculator", raw_input={"expression": "1+1"})
    reg.invoke(agent="data_analysis", task_id="t", tool_name="calculator", raw_input={"expression": "1+1"})
    with pytest.raises(RateLimitExceeded):
        reg.invoke(agent="data_analysis", task_id="t", tool_name="calculator", raw_input={"expression": "1+1"})


def test_code_exec_runs_python_in_sandbox():
    reg = get_registry()
    out = reg.invoke(agent="code_exec", task_id="t", tool_name="code_exec", raw_input={"code": "print(1+1)"})
    assert out["stdout"].strip() == "2"
    assert out["exit_code"] == 0


def test_code_exec_hints_when_code_forgets_to_print():
    """Regression test: python -c is a plain script, not a REPL -- a bare
    trailing expression produces no stdout. The tool used to return this
    silently, which caused the code_exec specialist to loop forever
    rewriting "broken" code that actually ran fine (see specialists/base.py
    MAX_TOOL_ITERATIONS failures). The tool should now hint at the mistake.
    """
    reg = get_registry()
    out = reg.invoke(agent="code_exec", task_id="t", tool_name="code_exec", raw_input={"code": "1 + 1"})
    assert out["stdout"] == ""
    assert out["exit_code"] == 0
    assert "print(...)" in out["stderr"]


def test_rate_limit_is_thread_safe_under_concurrent_callers():
    """Regression test: the rate limiter's check-then-append sequence used
    to run unguarded, so concurrent callers on the same tool (e.g. two
    Streamlit sessions in different threads) could both read "under limit"
    before either recorded their call, letting the limit be exceeded. A
    barrier maximizes contention so the race would actually manifest
    without the lock.
    """
    import threading

    reg = get_registry()
    tool = reg.get("calculator")
    tool.rate_limit_per_minute = 5

    n_threads = 30
    barrier = threading.Barrier(n_threads)
    successes = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        try:
            reg.invoke(agent="data_analysis", task_id="t", tool_name="calculator", raw_input={"expression": "1+1"})
            with lock:
                successes.append(1)
        except RateLimitExceeded:
            pass

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert len(successes) == 5


def test_tool_invocations_are_traced():
    from orchestration.tracing.tracer import get_trace_tree

    reg = get_registry()
    reg.invoke(agent="data_analysis", task_id="t", tool_name="calculator", raw_input={"expression": "5*5"})
    spans = get_trace_tree("t")
    tool_spans = [s for s in spans if s["span_type"] == "tool_call" and s["name"] == "calculator"]
    assert len(tool_spans) == 1
    assert tool_spans[0]["status"] == "success"
