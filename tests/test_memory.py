from orchestration.memory import long_term_memory, working_memory
from orchestration.db.connection import get_connection


def _task_row(task_id="t"):
    conn = get_connection()
    conn.execute("INSERT INTO tasks (id, user_id, request) VALUES (?, 'u', 'test')", (task_id,))
    conn.commit()


def test_working_memory_set_get_roundtrip():
    _task_row()
    working_memory.set_value("t", "plan", {"steps": [1, 2, 3]})
    assert working_memory.get_value("t", "plan") == {"steps": [1, 2, 3]}
    assert working_memory.get_value("t", "missing", default="fallback") == "fallback"


def test_working_memory_append_log_and_get_all():
    _task_row()
    working_memory.append_log("t", "errors", "first")
    working_memory.append_log("t", "errors", "second")
    assert working_memory.get_all("t") == {"errors": ["first", "second"]}


def test_working_memory_clear_scopes_to_task():
    _task_row("t1")
    _task_row("t2")
    working_memory.set_value("t1", "x", 1)
    working_memory.set_value("t2", "x", 2)
    working_memory.clear("t1")
    assert working_memory.get_all("t1") == {}
    assert working_memory.get_all("t2") == {"x": 2}


def test_long_term_memory_remember_and_recall(fake_embedding_function):
    long_term_memory.remember(user_id="u1", kind="preference", text="user prefers concise bullet-point summaries")
    long_term_memory.remember(user_id="u1", kind="fact", text="the sky is blue due to Rayleigh scattering")
    long_term_memory.remember(user_id="u2", kind="preference", text="user prefers long detailed reports")

    hits = long_term_memory.recall(user_id="u1", query="concise bullet-point summaries", n_results=5)
    assert len(hits) >= 1
    assert all(h.user_id == "u1" for h in hits)
    assert any("concise" in h.text for h in hits)


def test_long_term_memory_recall_reinforces_importance(fake_embedding_function):
    long_term_memory.remember(user_id="u1", kind="fact", text="quarterly revenue grew twelve percent")
    before = long_term_memory.dashboard_snapshot("u1")[0]["importance"]

    long_term_memory.recall(user_id="u1", query="quarterly revenue growth", n_results=1)

    after = long_term_memory.dashboard_snapshot("u1")[0]["importance"]
    assert after > before


def test_long_term_memory_delete_removes_all_user_memories(fake_embedding_function):
    long_term_memory.remember(user_id="u1", kind="fact", text="fact one")
    long_term_memory.remember(user_id="u1", kind="fact", text="fact two")
    long_term_memory.remember(user_id="u2", kind="fact", text="unrelated fact")

    deleted = long_term_memory.delete_user_memories("u1")

    assert deleted == 2
    assert long_term_memory.dashboard_snapshot("u1") == []
    assert len(long_term_memory.dashboard_snapshot("u2")) == 1


def test_long_term_memory_decay_expires_old_low_importance_memories(fake_embedding_function, monkeypatch):
    from datetime import datetime, timedelta

    long_term_memory.remember(user_id="u1", kind="fact", text="a one-off irrelevant detail")
    conn = get_connection()
    old_date = (datetime.utcnow() - timedelta(days=200)).isoformat()
    conn.execute(
        "UPDATE long_term_memory_meta SET importance = 0.01, created_at = ?, last_accessed_at = ? WHERE user_id = 'u1'",
        (old_date, old_date),
    )
    conn.commit()

    result = long_term_memory.decay_and_expire("u1")

    assert result["expired"] == 1
    assert long_term_memory.dashboard_snapshot("u1") == []
