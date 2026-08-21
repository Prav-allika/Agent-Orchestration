"""Gradio entry point for Hugging Face Spaces.

Full functional parity with the Streamlit console (ui/app.py + ui/pages/):
task submission, the human-in-the-loop approval queue (with the ask-the-
agent chat panel), the trace explorer, and the memory dashboard. Both UIs
share the same backend (orchestration.*) untouched -- this is a UI-layer
port, not a rewrite of the system.

Runs on Spaces' free ZeroGPU tier for hosting (see README's Deploying
section for why) even though nothing here ever touches a GPU: the app only
calls the OpenAI API over HTTP, so no @spaces.GPU-decorated function exists
anywhere in this file.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr

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

STATUS_ICON = {"success": "✅", "failure": "❌", "pending": "⏳", "escalated": "🙋"}


# --------------------------------------------------------------------- Tab 1: Task Console

def _recent_tasks_rows():
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, user_id, request, status, created_at FROM tasks ORDER BY created_at DESC LIMIT 25"
    ).fetchall()
    return [[r["id"], r["user_id"], r["request"][:80], r["status"], r["created_at"]] for r in rows]


def submit_task(user_id: str, request: str):
    if not request.strip():
        return "⚠️ Enter a task request first.", _recent_tasks_rows()
    try:
        result = start_task(user_id=user_id or "demo_user", request=request)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI instead of a raw traceback page
        return f"❌ Task submission failed: {exc}", _recent_tasks_rows()

    if result.status == "awaiting_approval":
        reason = result.interrupt.get("reason", "escalation triggered") if result.interrupt else "escalation triggered"
        msg = f"⏸️ Task **{result.task_id}** paused for human review ({reason}). Open the **Approval Queue** tab to resolve it."
    elif result.status == "completed":
        msg = f"✅ Task **{result.task_id}** completed.\n\n{result.final_output or '*(no output)*'}"
    else:
        msg = f"❌ Task **{result.task_id}** ended with status `{result.status}`.\n\n{result.final_output or ''}"
    return msg, _recent_tasks_rows()


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
            return dd, f"⚠️ Invalid plan JSON: {exc}", raw, chat, edited_text
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

    lines: list[str] = []

    def walk(span: dict, depth: int) -> None:
        icon = STATUS_ICON.get(span["status"], "•")
        indent = "  " * depth
        lines.append(
            f"{indent}- {icon} `{span['agent']}` · {span['span_type']} · **{span['name']}** "
            f"({span['latency_ms'] or 0:.0f}ms, ${span['cost_usd'] or 0:.4f})"
        )
        if span["error"]:
            lines.append(f"{indent}  > ⚠️ {span['error']}")
        for child in children.get(span["id"], []):
            walk(child, depth + 1)

    for root in children.get(None, []):
        walk(root, 0)
    return "\n".join(lines) or "*(no spans)*"


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
        return "", "*(no task selected)*", [], "", [], [], {}

    summary = get_task_cost_summary(task_id)
    summary_md = (
        f"**Spans:** {summary.get('span_count') or 0}  |  **Cost:** ${(summary.get('cost_usd') or 0):.4f}  |  "
        f"**Tokens:** {summary.get('tokens_prompt') or 0}p/{summary.get('tokens_completion') or 0}c  |  "
        f"**Wall time:** {(summary.get('latency_ms_total') or 0):.0f}ms  |  "
        f"**Failures/Escalations:** {summary.get('failures') or 0}/{summary.get('escalations') or 0}"
    )
    spans = get_trace_tree(task_id)
    tree_md = _render_tree(spans)

    quality = get_quality_metrics()
    quality_md = (
        f"**Task success rate:** {_fmt_rate(quality['task_success_rate'], 'completed')}  |  "
        f"**Reviewer first-pass rate:** {_fmt_rate(quality['reviewer_first_pass_rate'], 'passed')}  |  "
        f"**Human override rate:** {_fmt_rate(quality['human_override_rate'], 'overridden')}"
    )

    agg = get_aggregate_stats()
    by_agent_rows = [[r["agent"], r["spans"], round(r["cost_usd"] or 0, 4), round(r["latency_ms"] or 0, 1)] for r in agg["by_agent"]]
    by_tool_rows = [[r["name"], r["calls"], r["successes"], round(r["avg_latency_ms"] or 0, 1)] for r in agg["by_tool"]]

    return summary_md, tree_md, spans, quality_md, by_agent_rows, by_tool_rows, agg["totals"]


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

with gr.Blocks(title="Agent Orchestration Console") as demo:
    gr.Markdown("# 🧭 Agent Orchestration Console")
    gr.Markdown("Supervisor → specialists → reviewer, with memory and human-in-the-loop escalation.")

    with gr.Tabs():
        with gr.Tab("Task Console"):
            with gr.Row():
                user_id_in = gr.Textbox(label="User ID", value="demo_user")
            request_in = gr.Textbox(label="Task request", lines=4,
                                     placeholder="e.g. Research the current state of small modular reactors and "
                                                  "write a 300-word summary with at least 3 cited sources.")
            submit_btn = gr.Button("Submit Task", variant="primary")
            submit_out = gr.Markdown()
            gr.Markdown("### Recent tasks")
            recent_tasks_df = gr.Dataframe(headers=["id", "user_id", "request", "status", "created_at"], interactive=False)

            submit_btn.click(submit_task, [user_id_in, request_in], [submit_out, recent_tasks_df])
            demo.load(_recent_tasks_rows, None, recent_tasks_df)

        with gr.Tab("Approval Queue") as approval_tab:
            gr.Markdown("## 🙋 Human-in-the-Loop Approval Queue")
            refresh_btn = gr.Button("🔄 Refresh queue")
            pending_dd = gr.Dropdown(label="Pending escalations", choices=[])
            with gr.Row():
                with gr.Column(scale=3):
                    detail_md = gr.Markdown()
                    raw_json = gr.JSON(label="Raw context")
                    gr.Markdown("#### Ask the agent about this decision")
                    chatbot = gr.Chatbot(height=250)
                    chat_in = gr.Textbox(label="Question", placeholder="Ask a clarifying question before deciding...")
                    chat_send = gr.Button("Send")
                with gr.Column(scale=2):
                    gr.Markdown("### Decision")
                    approve_btn = gr.Button("✅ Approve", variant="primary")
                    reject_notes = gr.Textbox(label="Rejection notes")
                    reject_btn = gr.Button("❌ Reject")
                    modify_text = gr.Textbox(label="Edit plan JSON / specialist output", lines=8)
                    modify_btn = gr.Button("✏️ Approve modified")
                    takeover_text = gr.Textbox(label="Your output (take over)", lines=5)
                    takeover_btn = gr.Button("🧑‍💻 Take Over")

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
            gr.Markdown("## 🔬 Trace Explorer")
            trace_refresh_btn = gr.Button("🔄 Refresh tasks")
            task_dd = gr.Dropdown(label="Task", choices=[])
            summary_md = gr.Markdown()
            gr.Markdown("### Execution tree")
            tree_md = gr.Markdown()
            with gr.Accordion("Raw spans (JSON)", open=False):
                spans_json = gr.JSON()
            gr.Markdown("### Quality metrics (all tasks)")
            quality_md = gr.Markdown()
            gr.Markdown("### Aggregate stats (all tasks)")
            with gr.Row():
                by_agent_df = gr.Dataframe(headers=["agent", "spans", "cost_usd", "latency_ms"], label="Cost & latency by agent", interactive=False)
                by_tool_df = gr.Dataframe(headers=["name", "calls", "successes", "avg_latency_ms"], label="Tool usage", interactive=False)
            totals_json = gr.JSON(label="Totals")

            trace_outputs = [summary_md, tree_md, spans_json, quality_md, by_agent_df, by_tool_df, totals_json]
            trace_refresh_btn.click(refresh_task_choices, None, [task_dd, *trace_outputs])
            trace_tab.select(refresh_task_choices, None, [task_dd, *trace_outputs])
            task_dd.change(load_trace, task_dd, trace_outputs)

        with gr.Tab("Memory Dashboard"):
            gr.Markdown("## 🧠 Long-Term Memory Dashboard")
            gr.Markdown("What the system remembers about a given user, across all past tasks.")
            mem_user_id = gr.Textbox(label="User ID", value="demo_user")
            with gr.Row():
                decay_btn = gr.Button("Run decay / expiration pass")
                consolidate_btn = gr.Button("Consolidate similar task summaries")
                mem_confirm = gr.Checkbox(label="Confirm delete")
                delete_btn = gr.Button("🗑️ Delete ALL memories for this user")
            mem_status = gr.Markdown()
            mem_df = gr.Dataframe(
                headers=["kind", "importance", "access_count", "created_at", "last_accessed_at", "text"], interactive=False
            )

            decay_btn.click(do_decay, mem_user_id, [mem_status, mem_df])
            consolidate_btn.click(do_consolidate, mem_user_id, [mem_status, mem_df])
            delete_btn.click(do_delete, [mem_user_id, mem_confirm], [mem_status, mem_df])
            mem_user_id.change(_memory_rows, mem_user_id, mem_df)


if __name__ == "__main__":
    demo.launch()
