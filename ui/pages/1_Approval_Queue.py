import _bootstrap  # noqa: F401

import json

import streamlit as st

from orchestration.graph_runner import resume_task
from orchestration.hitl import approval_queue
from orchestration.hitl.chat import answer_question
from orchestration.schemas import ExecutionPlan

st.set_page_config(page_title="Approval Queue", page_icon="🙋", layout="wide")
st.title("🙋 Human-in-the-Loop Approval Queue")

pending = approval_queue.list_pending()
if not pending:
    st.info("Nothing pending review.")
    st.stop()

labels = [f"{r.id} · {r.level} · {r.trigger_reason[:50]}" for r in pending]
choice = st.selectbox("Pending escalations", options=range(len(pending)), format_func=lambda i: labels[i])
req = pending[choice]

col_context, col_decision = st.columns([3, 2])

with col_context:
    st.subheader(f"Task `{req.task_id}`")
    st.markdown(f"**Level:** `{req.level}`  \n**Trigger:** {req.trigger_reason}")

    if req.level == "approve_plan":
        plan = ExecutionPlan.model_validate(req.context["plan"])
        st.markdown(f"**Confidence:** {plan.confidence:.2f}  \n**Sensitive:** {plan.is_sensitive}")
        st.markdown(f"**Reasoning:** {plan.reasoning}")
        st.markdown("**Subtasks:**")
        for st_ in plan.subtasks:
            st.markdown(f"- `{st_.id}` [{st_.specialist}] {st_.description} (depends on: {st_.depends_on or 'none'})")
        if req.context.get("memories"):
            with st.expander("Memory used for this plan"):
                for m in req.context["memories"]:
                    st.markdown(f"- *(importance {m['importance']:.2f})* {m['text']}")
    else:
        subtask = req.context.get("subtask", {})
        st.markdown(f"**Subtask:** `{subtask.get('id')}` [{subtask.get('specialist')}] {subtask.get('description')}")
        st.markdown(f"**Retry count:** {req.context.get('retry_count')}  |  **Review retries:** {req.context.get('review_retries')}")
        if req.context.get("last_error"):
            st.error(f"Last error: {req.context['last_error']}")
        if req.context.get("last_review"):
            st.json(req.context["last_review"])
        if req.context.get("attempted_output"):
            with st.expander("Attempted output"):
                st.markdown(req.context["attempted_output"])

    with st.expander("Raw context"):
        st.json(req.context)

    st.markdown("#### Ask the agent about this decision")
    for msg in approval_queue.get_chat_messages(req.id):
        with st.chat_message("user" if msg["role"] == "human" else "assistant"):
            st.write(msg["content"])
    question = st.chat_input("Ask a clarifying question before deciding...")
    if question:
        approval_queue.add_chat_message(req.id, role="human", content=question)
        answer = answer_question(context=req.context, history=approval_queue.get_chat_messages(req.id), question=question)
        approval_queue.add_chat_message(req.id, role="agent", content=answer)
        st.rerun()

with col_decision:
    st.subheader("Decision")

    st.caption("The graph records the resolution on the approval request itself once resumed.")

    if st.button("✅ Approve", use_container_width=True):
        resume_task(task_id=req.task_id, resume_value={"decision": "approve"})
        st.rerun()

    with st.form(f"reject_{req.id}"):
        notes = st.text_area("Rejection notes")
        if st.form_submit_button("❌ Reject", use_container_width=True):
            resume_task(task_id=req.task_id, resume_value={"decision": "reject", "notes": notes})
            st.rerun()

    if req.level == "approve_plan":
        with st.form(f"modify_plan_{req.id}"):
            edited = st.text_area("Edit plan JSON", value=json.dumps(req.context["plan"], indent=2), height=300)
            if st.form_submit_button("✏️ Approve modified plan", use_container_width=True):
                try:
                    new_plan = ExecutionPlan.model_validate(json.loads(edited))
                except Exception as exc:
                    st.error(f"Invalid plan: {exc}")
                else:
                    resume_task(task_id=req.task_id, resume_value={"decision": "modify", "plan": new_plan.model_dump()})
                    st.rerun()
    else:
        with st.form(f"modify_action_{req.id}"):
            edited_output = st.text_area(
                "Edit the specialist's output directly",
                value=req.context.get("attempted_output", "") or "", height=200,
            )
            if st.form_submit_button("✏️ Approve modified output", use_container_width=True):
                resume_task(task_id=req.task_id, resume_value={"decision": "modify", "output": edited_output})
                st.rerun()

    with st.form(f"takeover_{req.id}"):
        st.caption("Take over: you provide the output directly, agents stand down for this step.")
        takeover_output = st.text_area("Your output", height=150)
        if st.form_submit_button("🧑‍💻 Take Over", use_container_width=True):
            resume_task(task_id=req.task_id, resume_value={"decision": "take_over", "output": takeover_output})
            st.rerun()
