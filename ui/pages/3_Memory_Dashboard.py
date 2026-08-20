import _bootstrap  # noqa: F401

import streamlit as st

from orchestration.memory import long_term_memory

st.set_page_config(page_title="Memory Dashboard", page_icon="🧠", layout="wide")
st.title("🧠 Long-Term Memory Dashboard")
st.caption("What the system remembers about a given user, across all past tasks.")

user_id = st.text_input("User ID", value="demo_user")

col1, col2, col3 = st.columns(3)
if col1.button("Run decay / expiration pass"):
    result = long_term_memory.decay_and_expire(user_id)
    st.success(f"Decayed {result['decayed']} memories, expired {result['expired']}.")

if col2.button("Consolidate similar task summaries"):
    result = long_term_memory.consolidate(user_id)
    st.success(f"Merged {result['merged_groups']} group(s) of similar memories.")

confirm = col3.checkbox("Confirm delete")
if col3.button("🗑️ Delete ALL memories for this user", disabled=not confirm):
    count = long_term_memory.delete_user_memories(user_id)
    st.success(f"Deleted {count} memories for '{user_id}'.")

st.divider()
snapshot = long_term_memory.dashboard_snapshot(user_id)
if not snapshot:
    st.info(f"No long-term memories stored for '{user_id}' yet -- complete a task first.")
else:
    import pandas as pd

    df = pd.DataFrame(snapshot)
    df["text"] = df["text"].str.slice(0, 150)
    st.dataframe(df, use_container_width=True, hide_index=True)
