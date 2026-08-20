import _bootstrap  # noqa: F401

import json
from collections import defaultdict

import streamlit as st

from orchestration.db.connection import get_connection
from orchestration.tracing.tracer import get_aggregate_stats, get_quality_metrics, get_task_cost_summary, get_trace_tree

st.set_page_config(page_title="Trace Explorer", page_icon="🔬", layout="wide")
st.title("🔬 Trace Explorer")

STATUS_ICON = {"success": "✅", "failure": "❌", "pending": "⏳", "escalated": "🙋"}

conn = get_connection()
tasks = conn.execute("SELECT id, request, status, created_at FROM tasks ORDER BY created_at DESC LIMIT 50").fetchall()

if not tasks:
    st.info("No tasks yet.")
    st.stop()

labels = [f"{t['id']} · {t['status']} · {t['request'][:50]}" for t in tasks]
choice = st.selectbox("Task", options=range(len(tasks)), format_func=lambda i: labels[i])
task_id = tasks[choice]["id"]

summary = get_task_cost_summary(task_id)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Spans", summary.get("span_count") or 0)
c2.metric("Cost (USD)", f"${(summary.get('cost_usd') or 0):.4f}")
c3.metric("Tokens (prompt/compl.)", f"{summary.get('tokens_prompt') or 0}/{summary.get('tokens_completion') or 0}")
c4.metric("Wall time (ms)", f"{(summary.get('latency_ms_total') or 0):.0f}")
c5.metric("Failures / Escalations", f"{summary.get('failures') or 0} / {summary.get('escalations') or 0}")

st.divider()
st.subheader("Execution tree")

spans = get_trace_tree(task_id)
children: dict[str | None, list[dict]] = defaultdict(list)
for s in spans:
    children[s["parent_span_id"]].append(s)


def render(span: dict, depth: int = 0):
    icon = STATUS_ICON.get(span["status"], "•")
    label = f"{icon} `{span['agent']}` · {span['span_type']} · **{span['name']}** ({span['latency_ms'] or 0:.0f}ms, ${span['cost_usd'] or 0:.4f})"
    with st.expander(label, expanded=(depth == 0 and span["status"] == "failure")):
        if span["error"]:
            st.error(span["error"])
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Input")
            st.json(json.loads(span["input_json"]) if span["input_json"] else {})
        with col_b:
            st.caption("Output")
            st.json(json.loads(span["output_json"]) if span["output_json"] else {})
        st.caption(
            f"tokens: {span['tokens_prompt']}p / {span['tokens_completion']}c · started {span['started_at']} · ended {span['ended_at']}"
        )
        for child in children.get(span["id"], []):
            render(child, depth + 1)


for root in children.get(None, []):
    render(root)

st.divider()
st.subheader("Quality metrics (all tasks)")
st.caption(
    "Correctness signals, as distinct from cost/latency below -- these answer "
    "\"is the system doing the right thing\", not just \"did it run\"."
)
quality = get_quality_metrics()


def _fmt_rate(bucket: dict, numerator_label: str) -> str:
    if bucket["rate"] is None:
        return "no data yet"
    return f"{bucket['rate']:.0%} ({bucket[numerator_label]}/{bucket['total']})"


q1, q2, q3 = st.columns(3)
q1.metric("Task success rate", _fmt_rate(quality["task_success_rate"], "completed"))
q2.metric(
    "Reviewer first-pass rate", _fmt_rate(quality["reviewer_first_pass_rate"], "passed"),
    help="Fraction of subtasks the reviewer accepted on the FIRST attempt, one row per subtask "
         "(retries don't inflate the denominator). Low = specialists are getting it wrong often.",
)
q3.metric(
    "Human override rate", _fmt_rate(quality["human_override_rate"], "overridden"),
    help="Fraction of resolved escalations where the human did NOT simply approve the agent's "
         "proposal as-is. Near-zero could mean the system is genuinely good, or that the reviewer "
         "isn't being critical -- read this alongside who's actually resolving the queue.",
)
if quality["human_override_rate"]["breakdown"]:
    st.caption(f"Resolution breakdown: {quality['human_override_rate']['breakdown']}")

st.divider()
st.subheader("Aggregate stats (all tasks)")
agg = get_aggregate_stats()

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Cost & latency by agent**")
    st.dataframe(agg["by_agent"], use_container_width=True, hide_index=True)
with col2:
    st.markdown("**Tool usage**")
    st.dataframe(agg["by_tool"], use_container_width=True, hide_index=True)

st.markdown("**Totals**")
st.json(agg["totals"])
