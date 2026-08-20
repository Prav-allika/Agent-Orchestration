import _bootstrap  # noqa: F401  -- must run before orchestration imports

import streamlit as st

from orchestration.db.connection import get_connection
from orchestration.graph_runner import start_task

st.set_page_config(page_title="Agent Orchestration Console", page_icon="🧭", layout="wide")
st.title("🧭 Agent Orchestration Console")
st.caption("Supervisor → specialists → reviewer, with memory and human-in-the-loop escalation.")

with st.form("submit_task"):
    user_id = st.text_input("User ID", value="demo_user")
    request = st.text_area(
        "Task request", height=120,
        placeholder="e.g. Research the current state of small modular reactors and write a 300-word "
                    "summary with at least 3 cited sources.",
    )
    submitted = st.form_submit_button("Submit Task", type="primary")

if submitted:
    if not request.strip():
        st.error("Enter a task request first.")
    else:
        with st.spinner("Running supervisor → specialists → reviewer... this calls the LLM several times."):
            result = start_task(user_id=user_id, request=request)
        st.session_state["last_task_id"] = result.task_id

        if result.status == "awaiting_approval":
            st.warning(
                f"Task **{result.task_id}** paused for human review "
                f"({result.interrupt.get('reason', 'escalation triggered')}). "
                "Open **Approval Queue** in the sidebar to resolve it."
            )
        elif result.status == "completed":
            st.success(f"Task **{result.task_id}** completed.")
            st.markdown(result.final_output or "*(no output)*")
        else:
            st.error(f"Task **{result.task_id}** ended with status `{result.status}`.")
            if result.final_output:
                st.markdown(result.final_output)

st.divider()
st.subheader("Recent tasks")
conn = get_connection()
rows = conn.execute(
    "SELECT id, user_id, request, status, created_at, completed_at FROM tasks ORDER BY created_at DESC LIMIT 25"
).fetchall()
if rows:
    import pandas as pd

    df = pd.DataFrame([dict(r) for r in rows])
    df["request"] = df["request"].str.slice(0, 80)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No tasks yet -- submit one above.")
