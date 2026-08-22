"""Gradio entry point for Hugging Face Spaces.

Full functional parity with the Streamlit console (ui/app.py + ui/pages/):
task submission, the human-in-the-loop approval queue (with the ask-the-
agent chat panel), the trace explorer, and the memory dashboard. Both UIs
share the same backend (orchestration.*) untouched -- this is a UI-layer
port, not a rewrite of the system.

Runs on Spaces' free ZeroGPU tier for hosting (see README's Deploying
section for why) even though nothing here ever touches a GPU: the app only
calls the OpenAI API over HTTP. HF's ZeroGPU Space validation requires at
least one @spaces.GPU-decorated function to be present at startup, though
-- see _zerogpu_keepalive() below for the (unused) function that satisfies
that check.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr
import markdown as md
import spaces

from orchestration.db.connection import get_connection
from orchestration.graph_runner import resume_task, start_task
from orchestration.hitl import approval_queue
from orchestration.hitl.chat import answer_question
from orchestration.memory import long_term_memory
from orchestration.schemas import ExecutionPlan
from orchestration.tracing.tracer import (
    get_aggregate_stats,
    get_quality_metrics,
    get_task_cost_summary,
    get_trace_tree,
)

STATUS_LABEL = {"success": "success", "failure": "failure", "pending": "pending", "escalated": "escalated"}


@spaces.GPU(duration=1)
def _zerogpu_keepalive() -> None:
    """Never called. Exists solely so Hugging Face's ZeroGPU Space startup
    validation ("No @spaces.GPU function detected") passes -- this app has
    no real GPU workload, it only calls the OpenAI API over HTTP. The
    decorator itself is a documented no-op outside an actual ZeroGPU
    runtime, so this is harmless locally and in any other deployment.
    """
    return None


# --------------------------------------------------------------------------------- Design system
#
# Dark navy editorial base (inspired by clean, serif-headlined marketing pages) with a warm
# amber/peach gradient accent used consistently for badges, primary buttons, and card borders.
# Icons are inline SVG (Feather-style line icons, MIT-licensed shapes, no external requests) so
# every emoji in the previous version becomes a real icon instead of relying on font emoji glyphs.

def _icon(paths: str, size: int = 15) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:block">{paths}</svg>'
    )


ICON_COMPASS = _icon('<circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>', 30)
ICON_LIST = _icon('<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>')
ICON_SHIELD = _icon('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>')
ICON_ACTIVITY = _icon('<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>')
ICON_DATABASE = _icon('<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>')
ICON_CHAT = _icon('<path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>')
ICON_SLIDERS = _icon('<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>')
ICON_BAR_CHART = _icon('<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>')
ICON_CLOCK = _icon('<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>')


def _badge_header(icon: str, title: str, subtitle: str | None = None) -> str:
    sub = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    return f'<div class="badge-row"><span class="badge">{icon}{title}</span></div>{sub}'


def _hero() -> str:
    return f"""
    <div class="hero">
      <div style="display:flex; align-items:center; gap:16px;">
        <div class="hero-mark">{ICON_COMPASS}</div>
        <div>
          <div class="hero-title">Agent Orchestration Console</div>
          <div class="hero-subtitle">Supervisor &rarr; specialists &rarr; reviewer, with memory and human-in-the-loop escalation.</div>
        </div>
      </div>
    </div>
    """


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #10141d;
    --bg-alt: #161c29;
    --bg-card: #1b2231;
    --accent-1: #f3a462;
    --accent-2: #c96a34;
    --accent-soft: rgba(243, 164, 98, 0.14);
    --text-main: #f3ede4;
    --text-muted: #98a2b8;
    --border-soft: #2a3247;
    --success: #6fcf97;
    --danger: #eb5757;
    --pending: #98a2b8;
}

.gradio-container { background: var(--bg) !important; font-family: 'Inter', ui-sans-serif, sans-serif !important; }
.gradio-container, .gradio-container p, .gradio-container span, .gradio-container li { color: var(--text-main); }
.gradio-container .block { background: var(--bg-alt) !important; border-color: var(--border-soft) !important; }
.gradio-container label span, .gradio-container .label-wrap span { color: var(--text-muted) !important; }
.gradio-container input, .gradio-container textarea, .gradio-container select {
    background: var(--bg-card) !important; color: var(--text-main) !important; border-color: var(--border-soft) !important;
}

.hero {
    padding: 26px 30px; margin-bottom: 4px; border-radius: 18px;
    background: linear-gradient(135deg, #1a2131 0%, #10141d 100%);
    border: 1px solid var(--border-soft);
}
.hero-mark { color: var(--accent-1); }
.hero-title {
    font-family: 'Fraunces', serif !important; font-size: 2.1rem !important; font-weight: 600 !important;
    margin: 0 0 6px 0 !important; color: var(--accent-1) !important;
}
.hero-subtitle { color: var(--text-muted); font-size: 0.98rem; margin: 0; }

.badge-row { display: flex; align-items: center; gap: 10px; margin: 6px 0 4px 0; }
.badge {
    display: inline-flex; align-items: center; gap: 8px; padding: 7px 16px; border-radius: 999px;
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2)); color: #2a1608;
    font-weight: 700; font-size: 0.74rem; letter-spacing: 0.06em; text-transform: uppercase; white-space: nowrap;
}
.section-subtitle { color: var(--text-muted); font-size: 0.88rem; margin: 2px 0 10px 2px; }

.content-box {
    background: var(--bg-card) !important; border-left: 4px solid var(--accent-1) !important;
    border-radius: 12px !important; padding: 16px 20px !important;
}
.content-box p, .content-box li, .content-box strong, .content-box h1, .content-box h2, .content-box h3 { color: var(--text-main) !important; }
.content-box code { color: var(--accent-1); }

.status-card { border-radius: 12px; padding: 16px 20px; border-left: 4px solid var(--pending); background: var(--bg-card); }
.status-card.ok { border-left-color: var(--success); }
.status-card.warn { border-left-color: var(--accent-1); }
.status-card.err { border-left-color: var(--danger); }
.status-label { font-weight: 700; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em; display: block; margin-bottom: 8px; }
.status-card.ok .status-label { color: var(--success); }
.status-card.warn .status-label { color: var(--accent-1); }
.status-card.err .status-label { color: var(--danger); }
.status-body { color: var(--text-main); line-height: 1.6; }
.status-body p { margin: 0 0 10px 0; }
.status-body p:last-child { margin-bottom: 0; }
.status-body ul, .status-body ol { margin: 0 0 10px 20px; padding: 0; }
.status-body a { color: var(--accent-1); }
.status-body strong { color: #fbeee0; }

.trace-tree { font-size: 0.88rem; line-height: 2; }
.trace-node { display: flex; align-items: center; gap: 9px; }
.status-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.status-dot.success { background: var(--success); }
.status-dot.failure { background: var(--danger); }
.status-dot.pending { background: var(--pending); }
.status-dot.escalated { background: var(--accent-1); }
.trace-agent { color: var(--accent-1); font-weight: 600; }
.trace-meta { color: var(--text-muted); font-size: 0.82rem; }
.trace-error { color: var(--danger); font-size: 0.82rem; margin: 0 0 4px 20px; }

.gradio-container button.primary {
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2)) !important;
    border: none !important; color: #2a1608 !important; font-weight: 700 !important;
}
.gradio-container button.secondary {
    background: transparent !important; border: 1px solid var(--accent-1) !important; color: var(--accent-1) !important;
}

.gradio-container .tab-nav button.selected { color: var(--accent-1) !important; border-color: var(--accent-1) !important; }
"""


# --------------------------------------------------------------------- Tab 1: Task Console

def _recent_tasks_rows():
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, user_id, request, status, created_at FROM tasks ORDER BY created_at DESC LIMIT 25"
    ).fetchall()
    return [[r["id"], r["user_id"], r["request"][:80], r["status"], r["created_at"]] for r in rows]


def _status_card(kind: str, label: str, body: str) -> str:
    # The agent's final_output is markdown (bold, links, bullet lists from the LLM) -- render it
    # properly instead of dumping raw ** and [text](url) syntax into a plain HTML div.
    body_html = md.markdown(body, extensions=["extra"])
    return f'<div class="status-card {kind}"><span class="status-label">{label}</span><div class="status-body">{body_html}</div></div>'


def submit_task(user_id: str, request: str):
    if not request.strip():
        return _status_card("warn", "Missing input", "Enter a task request first."), _recent_tasks_rows()
    try:
        result = start_task(user_id=user_id or "demo_user", request=request)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI instead of a raw traceback page
        return _status_card("err", "Submission failed", str(exc)), _recent_tasks_rows()

    if result.status == "awaiting_approval":
        reason = result.interrupt.get("reason", "escalation triggered") if result.interrupt else "escalation triggered"
        html = _status_card("warn", "Awaiting human review",
                             f"Task <b>{result.task_id}</b> paused ({reason}). Open the Approval Queue tab to resolve it.")
    elif result.status == "completed":
        html = _status_card("ok", f"Completed &middot; {result.task_id}", result.final_output or "(no output)")
    else:
        html = _status_card("err", f"{result.status} &middot; {result.task_id}", result.final_output or "")
    return html, _recent_tasks_rows()


# --------------------------------------------------------------------- Tab 2: Approval Queue

def _pending_choices():
    return [(f"{r.id} · {r.level} · {r.trigger_reason[:50]}", r.id) for r in approval_queue.list_pending()]


def _render_approval(approval_id: str | None):
    if not approval_id:
        return "Nothing pending review.", {}
    req = approval_queue.get(approval_id)
    if req is None:
        return "Not found (it may have just been resolved).", {}

    lines = [f"### Task `{req.task_id}`", f"**Level:** `{req.level}`", f"**Trigger:** {req.trigger_reason}"]
    if req.level == "approve_plan":
        plan = ExecutionPlan.model_validate(req.context["plan"])
        lines.append(f"**Confidence:** {plan.confidence:.2f}  |  **Sensitive:** {plan.is_sensitive}")
        lines.append(f"**Reasoning:** {plan.reasoning}")
        lines.append("**Subtasks:**")
        for st in plan.subtasks:
            lines.append(f"- `{st.id}` [{st.specialist}] {st.description} (depends on: {st.depends_on or 'none'})")
        if req.context.get("memories"):
            lines.append("**Memory used for this plan:**")
            for m in req.context["memories"]:
                lines.append(f"- *(importance {m['importance']:.2f})* {m['text']}")
    else:
        subtask = req.context.get("subtask", {})
        lines.append(f"**Subtask:** `{subtask.get('id')}` [{subtask.get('specialist')}] {subtask.get('description')}")
        lines.append(f"**Retry count:** {req.context.get('retry_count')}  |  **Review retries:** {req.context.get('review_retries')}")
        if req.context.get("last_error"):
            lines.append(f"**Last error:** {req.context['last_error']}")
        if req.context.get("last_review"):
            lines.append(f"**Last review:** {json.dumps(req.context['last_review'])}")
        if req.context.get("attempted_output"):
            lines.append(f"**Attempted output:**\n\n{req.context['attempted_output']}")
    return "\n\n".join(lines), req.context


def _chat_history(approval_id: str | None):
    if not approval_id:
        return []
    return [{"role": "user" if m["role"] == "human" else "assistant", "content": m["content"]}
            for m in approval_queue.get_chat_messages(approval_id)]


def refresh_queue():
    choices = _pending_choices()
    value = choices[0][1] if choices else None
    detail, raw = _render_approval(value)
    return gr.update(choices=choices, value=value), detail, raw, _chat_history(value)


def on_select_approval(approval_id: str | None):
    detail, raw = _render_approval(approval_id)
    return detail, raw, _chat_history(approval_id)


def do_approve(approval_id: str | None):
    if approval_id:
        req = approval_queue.get(approval_id)
        if req:
            resume_task(task_id=req.task_id, resume_value={"decision": "approve"})
    return refresh_queue()


def do_reject(approval_id: str | None, notes: str):
    if approval_id:
        req = approval_queue.get(approval_id)
        if req:
            resume_task(task_id=req.task_id, resume_value={"decision": "reject", "notes": notes})
    return (*refresh_queue(), "")


def do_modify(approval_id: str | None, edited_text: str):
    if not approval_id:
        return (*refresh_queue(), edited_text)
    req = approval_queue.get(approval_id)
    if req is None:
        return (*refresh_queue(), edited_text)

    if req.level == "approve_plan":
        try:
            new_plan = ExecutionPlan.model_validate(json.loads(edited_text))
        except Exception as exc:
            dd, _, raw, chat = refresh_queue()
            return dd, f"**Invalid plan JSON:** {exc}", raw, chat, edited_text
        resume_task(task_id=req.task_id, resume_value={"decision": "modify", "plan": new_plan.model_dump()})
    else:
        resume_task(task_id=req.task_id, resume_value={"decision": "modify", "output": edited_text})
    return (*refresh_queue(), "")


def do_takeover(approval_id: str | None, output_text: str):
    if approval_id:
        req = approval_queue.get(approval_id)
        if req:
            resume_task(task_id=req.task_id, resume_value={"decision": "take_over", "output": output_text})
    return (*refresh_queue(), "")


def send_chat(approval_id: str | None, question: str, history: list):
    if not approval_id or not question.strip():
        return history, ""
    req = approval_queue.get(approval_id)
    if req is None:
        return history, ""
    approval_queue.add_chat_message(approval_id, role="human", content=question)
    answer = answer_question(context=req.context, history=approval_queue.get_chat_messages(approval_id), question=question)
    approval_queue.add_chat_message(approval_id, role="agent", content=answer)
    return _chat_history(approval_id), ""


# --------------------------------------------------------------------- Tab 3: Trace Explorer

def _task_choices():
    conn = get_connection()
    rows = conn.execute("SELECT id, request, status, created_at FROM tasks ORDER BY created_at DESC LIMIT 50").fetchall()
    return [(f"{r['id']} · {r['status']} · {r['request'][:50]}", r["id"]) for r in rows]


def _render_tree(spans: list[dict]) -> str:
    children: dict[str | None, list[dict]] = defaultdict(list)
    for s in spans:
        children[s["parent_span_id"]].append(s)

    lines: list[str] = ['<div class="trace-tree">']

    def walk(span: dict, depth: int) -> None:
        status = STATUS_LABEL.get(span["status"], "pending")
        indent = depth * 20
        lines.append(
            f'<div class="trace-node" style="margin-left:{indent}px">'
            f'<span class="status-dot {status}"></span>'
            f'<span class="trace-agent">{span["agent"]}</span>'
            f'<span class="trace-meta">&middot; {span["span_type"]} &middot;</span>'
            f'<b>{span["name"]}</b>'
            f'<span class="trace-meta">({span["latency_ms"] or 0:.0f}ms, ${span["cost_usd"] or 0:.4f})</span>'
            f'</div>'
        )
        if span["error"]:
            lines.append(f'<div class="trace-error" style="margin-left:{indent + 20}px">{span["error"]}</div>')
        for child in children.get(span["id"], []):
            walk(child, depth + 1)

    for root in children.get(None, []):
        walk(root, 0)
    lines.append("</div>")
    return "".join(lines) if len(lines) > 2 else '<div class="trace-tree">(no spans)</div>'


def _fmt_rate(bucket: dict, numerator_label: str) -> str:
    if bucket["rate"] is None:
        return "no data yet"
    return f"{bucket['rate']:.0%} ({bucket[numerator_label]}/{bucket['total']})"


def refresh_task_choices():
    choices = _task_choices()
    value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=value), *load_trace(value)


def load_trace(task_id: str | None):
    if not task_id:
        return "", '<div class="trace-tree">(no task selected)</div>', [], "", [], [], {}

    summary = get_task_cost_summary(task_id)
    summary_md = (
        f"**Spans:** {summary.get('span_count') or 0}  |  **Cost:** ${(summary.get('cost_usd') or 0):.4f}  |  "
        f"**Tokens:** {summary.get('tokens_prompt') or 0}p/{summary.get('tokens_completion') or 0}c  |  "
        f"**Wall time:** {(summary.get('latency_ms_total') or 0):.0f}ms  |  "
        f"**Failures/Escalations:** {summary.get('failures') or 0}/{summary.get('escalations') or 0}"
    )
    spans = get_trace_tree(task_id)
    tree_html = _render_tree(spans)

    quality = get_quality_metrics()
    quality_md = (
        f"**Task success rate:** {_fmt_rate(quality['task_success_rate'], 'completed')}  |  "
        f"**Reviewer first-pass rate:** {_fmt_rate(quality['reviewer_first_pass_rate'], 'passed')}  |  "
        f"**Human override rate:** {_fmt_rate(quality['human_override_rate'], 'overridden')}"
    )

    agg = get_aggregate_stats()
    by_agent_rows = [[r["agent"], r["spans"], round(r["cost_usd"] or 0, 4), round(r["latency_ms"] or 0, 1)] for r in agg["by_agent"]]
    by_tool_rows = [[r["name"], r["calls"], r["successes"], round(r["avg_latency_ms"] or 0, 1)] for r in agg["by_tool"]]

    return summary_md, tree_html, spans, quality_md, by_agent_rows, by_tool_rows, agg["totals"]


# --------------------------------------------------------------------- Tab 4: Memory Dashboard

def _memory_rows(user_id: str):
    snapshot = long_term_memory.dashboard_snapshot(user_id)
    return [[m["kind"], m["importance"], m["access_count"], m["created_at"], m["last_accessed_at"], m["text"][:150]] for m in snapshot]


def do_decay(user_id: str):
    result = long_term_memory.decay_and_expire(user_id)
    return f"Decayed {result['decayed']} memories, expired {result['expired']}.", _memory_rows(user_id)


def do_consolidate(user_id: str):
    result = long_term_memory.consolidate(user_id)
    return f"Merged {result['merged_groups']} group(s) of similar memories.", _memory_rows(user_id)


def do_delete(user_id: str, confirm: bool):
    if not confirm:
        return "Check 'Confirm delete' first.", _memory_rows(user_id)
    count = long_term_memory.delete_user_memories(user_id)
    return f"Deleted {count} memories for '{user_id}'.", _memory_rows(user_id)


# --------------------------------------------------------------------------------- Build UI

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
).set(
    body_background_fill="#10141d",
    body_background_fill_dark="#10141d",
    body_text_color="#f3ede4",
    body_text_color_dark="#f3ede4",
    background_fill_primary="#161c29",
    background_fill_primary_dark="#161c29",
    background_fill_secondary="#1b2231",
    background_fill_secondary_dark="#1b2231",
    block_background_fill="#161c29",
    block_background_fill_dark="#161c29",
    block_border_color="#2a3247",
    block_border_color_dark="#2a3247",
    block_label_text_color="#98a2b8",
    block_label_text_color_dark="#98a2b8",
    block_title_text_color="#f3ede4",
    block_title_text_color_dark="#f3ede4",
    panel_background_fill="#161c29",
    panel_background_fill_dark="#161c29",
    border_color_primary="#2a3247",
    border_color_primary_dark="#2a3247",
    input_background_fill="#1b2231",
    input_background_fill_dark="#1b2231",
    button_primary_background_fill="linear-gradient(135deg, #f3a462, #c96a34)",
    button_primary_background_fill_hover="linear-gradient(135deg, #f5b57d, #d97a40)",
    button_primary_text_color="#2a1608",
    button_secondary_background_fill="transparent",
    button_secondary_border_color="#f3a462",
    button_secondary_text_color="#f3a462",
)

with gr.Blocks(title="Agent Orchestration Console") as demo:
    gr.HTML(_hero())

    with gr.Tabs():
        with gr.Tab("Task Console") as task_tab:
            gr.HTML(_badge_header(ICON_LIST, "Task Console"))
            with gr.Group():
                user_id_in = gr.Textbox(label="User ID", value="demo_user")
                request_in = gr.Textbox(label="Task request", lines=4,
                                         placeholder="e.g. Research the current state of small modular reactors and "
                                                      "write a 300-word summary with at least 3 cited sources.")
                submit_btn = gr.Button("Submit Task", variant="primary")
            submit_out = gr.HTML()
            gr.HTML(_badge_header(ICON_CLOCK, "Recent Tasks"))
            recent_tasks_df = gr.Dataframe(headers=["id", "user_id", "request", "status", "created_at"], interactive=False)

            submit_btn.click(submit_task, [user_id_in, request_in], [submit_out, recent_tasks_df])
            demo.load(_recent_tasks_rows, None, recent_tasks_df)
            task_tab.select(_recent_tasks_rows, None, recent_tasks_df)
            # Genuine auto-refresh: a task can complete in the background (after an approval was
            # resolved elsewhere) while this tab stays open, so poll on an interval rather than only
            # refreshing on tab-select or task submission.
            gr.Timer(4).tick(_recent_tasks_rows, None, recent_tasks_df)

        with gr.Tab("Approval Queue") as approval_tab:
            gr.HTML(_badge_header(ICON_SHIELD, "Human-in-the-Loop Approval Queue"))
            refresh_btn = gr.Button("Refresh Queue", variant="secondary")
            pending_dd = gr.Dropdown(label="Pending escalations", choices=[])
            with gr.Row():
                with gr.Column(scale=3):
                    detail_md = gr.Markdown(elem_classes=["content-box"])
                    raw_json = gr.JSON(label="Raw context")
                    gr.HTML(_badge_header(ICON_CHAT, "Ask The Agent"))
                    chatbot = gr.Chatbot(height=250)
                    chat_in = gr.Textbox(label="Question", placeholder="Ask a clarifying question before deciding...")
                    chat_send = gr.Button("Send", variant="secondary")
                with gr.Column(scale=2):
                    gr.HTML(_badge_header(ICON_SLIDERS, "Decision"))
                    with gr.Group():
                        approve_btn = gr.Button("Approve", variant="primary")
                        reject_notes = gr.Textbox(label="Rejection notes")
                        reject_btn = gr.Button("Reject", variant="secondary")
                        modify_text = gr.Textbox(label="Edit plan JSON / specialist output", lines=8)
                        modify_btn = gr.Button("Approve Modified", variant="secondary")
                        takeover_text = gr.Textbox(label="Your output (take over)", lines=5)
                        takeover_btn = gr.Button("Take Over", variant="secondary")

            queue_outputs = [pending_dd, detail_md, raw_json, chatbot]
            refresh_btn.click(refresh_queue, None, queue_outputs)
            approval_tab.select(refresh_queue, None, queue_outputs)
            pending_dd.change(on_select_approval, pending_dd, [detail_md, raw_json, chatbot])
            approve_btn.click(do_approve, pending_dd, queue_outputs)
            reject_btn.click(do_reject, [pending_dd, reject_notes], [*queue_outputs, reject_notes])
            modify_btn.click(do_modify, [pending_dd, modify_text], [*queue_outputs, modify_text])
            takeover_btn.click(do_takeover, [pending_dd, takeover_text], [*queue_outputs, takeover_text])
            chat_send.click(send_chat, [pending_dd, chat_in, chatbot], [chatbot, chat_in])
            chat_in.submit(send_chat, [pending_dd, chat_in, chatbot], [chatbot, chat_in])

        with gr.Tab("Trace Explorer") as trace_tab:
            gr.HTML(_badge_header(ICON_ACTIVITY, "Trace Explorer"))
            trace_refresh_btn = gr.Button("Refresh Tasks", variant="secondary")
            task_dd = gr.Dropdown(label="Task", choices=[])
            summary_md = gr.Markdown()
            gr.HTML(_badge_header(ICON_BAR_CHART, "Execution Tree"))
            tree_html = gr.HTML(elem_classes=["content-box"])
            with gr.Accordion("Raw spans (JSON)", open=False):
                spans_json = gr.JSON()
            gr.HTML(_badge_header(ICON_ACTIVITY, "Quality Metrics"))
            quality_md = gr.Markdown(elem_classes=["content-box"])
            gr.HTML(_badge_header(ICON_BAR_CHART, "Aggregate Stats"))
            with gr.Row():
                by_agent_df = gr.Dataframe(headers=["agent", "spans", "cost_usd", "latency_ms"], label="Cost & latency by agent", interactive=False)
                by_tool_df = gr.Dataframe(headers=["name", "calls", "successes", "avg_latency_ms"], label="Tool usage", interactive=False)
            totals_json = gr.JSON(label="Totals")

            trace_outputs = [summary_md, tree_html, spans_json, quality_md, by_agent_df, by_tool_df, totals_json]
            trace_refresh_btn.click(refresh_task_choices, None, [task_dd, *trace_outputs])
            trace_tab.select(refresh_task_choices, None, [task_dd, *trace_outputs])
            task_dd.change(load_trace, task_dd, trace_outputs)

        with gr.Tab("Memory Dashboard"):
            gr.HTML(_badge_header(ICON_DATABASE, "Long-Term Memory Dashboard", "What the system remembers about a given user, across all past tasks."))
            mem_user_id = gr.Textbox(label="User ID", value="demo_user")
            with gr.Group():
                with gr.Row():
                    decay_btn = gr.Button("Run Decay / Expiration Pass", variant="secondary")
                    consolidate_btn = gr.Button("Consolidate Similar Summaries", variant="secondary")
                    mem_confirm = gr.Checkbox(label="Confirm delete")
                    delete_btn = gr.Button("Delete All Memories", variant="secondary")
            mem_status = gr.Markdown()
            mem_df = gr.Dataframe(
                headers=["kind", "importance", "access_count", "created_at", "last_accessed_at", "text"], interactive=False
            )

            decay_btn.click(do_decay, mem_user_id, [mem_status, mem_df])
            consolidate_btn.click(do_consolidate, mem_user_id, [mem_status, mem_df])
            delete_btn.click(do_delete, [mem_user_id, mem_confirm], [mem_status, mem_df])
            mem_user_id.change(_memory_rows, mem_user_id, mem_df)


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS)
