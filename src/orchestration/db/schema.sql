-- SQLite schema backing working memory, tracing, and the human-in-the-loop
-- approval queue. Long-term semantic memory lives in ChromaDB; the
-- long_term_memory_meta table here only tracks importance/expiration
-- bookkeeping that ChromaDB doesn't natively support.

CREATE TABLE IF NOT EXISTS tasks (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    request         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending/planning/running/awaiting_approval/completed/failed
    result          TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at    TEXT
);

CREATE TABLE IF NOT EXISTS execution_plans (
    id              TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL REFERENCES tasks(id),
    subtasks_json   TEXT NOT NULL,   -- serialized list[Subtask]
    confidence      REAL NOT NULL,
    reasoning       TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Short-term working memory: scoped to a single task, cleared on completion.
-- Stands in for the Redis store described in the spec; same access pattern
-- (namespaced key/value, cheap point reads/writes) so swapping the backend
-- later is a storage-layer change only.
CREATE TABLE IF NOT EXISTS working_memory (
    task_id         TEXT NOT NULL,
    key             TEXT NOT NULL,
    value_json      TEXT NOT NULL,
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (task_id, key)
);

CREATE TABLE IF NOT EXISTS long_term_memory_meta (
    id              TEXT PRIMARY KEY,     -- matches the ChromaDB document id
    user_id         TEXT NOT NULL,
    kind            TEXT NOT NULL,        -- task_summary/preference/fact/approach
    importance      REAL NOT NULL DEFAULT 1.0,
    access_count    INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    last_accessed_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at      TEXT
);

CREATE TABLE IF NOT EXISTS trace_spans (
    id              TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL REFERENCES tasks(id),
    parent_span_id  TEXT,
    agent           TEXT NOT NULL,        -- supervisor/research/data_analysis/writer/code_exec/reviewer/human
    span_type       TEXT NOT NULL,        -- plan/specialist_step/tool_call/review/escalation/synthesis
    name            TEXT NOT NULL,
    input_json      TEXT,
    output_json     TEXT,
    status          TEXT NOT NULL,        -- success/failure/escalated/pending
    error           TEXT,
    latency_ms      REAL,
    tokens_prompt   INTEGER DEFAULT 0,
    tokens_completion INTEGER DEFAULT 0,
    cost_usd        REAL DEFAULT 0,
    started_at      TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at        TEXT
);

CREATE INDEX IF NOT EXISTS idx_trace_spans_task ON trace_spans(task_id);

CREATE TABLE IF NOT EXISTS approval_requests (
    id              TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL REFERENCES tasks(id),
    level           TEXT NOT NULL,        -- notify/approve_action/approve_plan/take_over
    trigger_reason  TEXT NOT NULL,
    context_json    TEXT NOT NULL,        -- full packaged context for the reviewer
    proposed_action_json TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending/approved/rejected/modified/take_over/notified
    resolution_json TEXT,
    reviewer_notes  TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_approval_requests_status ON approval_requests(status);

CREATE TABLE IF NOT EXISTS approval_chat_messages (
    id              TEXT PRIMARY KEY,
    approval_id     TEXT NOT NULL REFERENCES approval_requests(id),
    role            TEXT NOT NULL,        -- human/agent
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Golden-dataset eval runs: LLM-as-judge correctness scoring, distinct from
-- trace_spans (which measures "did it run") -- this measures "was the
-- output actually right". eval_run_id groups all golden tasks scored in
-- one invocation of the eval harness, so results are comparable run-over-run.
CREATE TABLE IF NOT EXISTS eval_results (
    id              TEXT PRIMARY KEY,
    eval_run_id     TEXT NOT NULL,
    golden_task_id  TEXT NOT NULL,
    task_id         TEXT NOT NULL REFERENCES tasks(id),
    request         TEXT NOT NULL,
    final_output    TEXT,
    task_status     TEXT NOT NULL,        -- completed/failed
    judge_passed    INTEGER,              -- 0/1, NULL if judging failed
    judge_score     REAL,
    judge_json      TEXT,                 -- full JudgeResult, criterion-by-criterion
    cost_usd        REAL DEFAULT 0,
    latency_ms      REAL DEFAULT 0,
    escalation_count INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_eval_results_run ON eval_results(eval_run_id);
