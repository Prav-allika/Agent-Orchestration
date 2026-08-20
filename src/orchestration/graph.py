"""The LangGraph state machine: task intake -> planning -> (plan approval?)
-> specialist execution loop (subtask -> review -> [retry|escalate|next])
-> synthesis -> delivery.

Pause/resume for human-in-the-loop uses LangGraph's native interrupt()/
Command(resume=...) mechanism backed by a SQLite checkpointer (see
graph_runner.py), which is what makes "the system waits for approval...
before continuing" (spec Phase 3.2) work across process boundaries: the
demo/CLI can start a task, and the Streamlit review UI can resume it later
in a totally separate process, because both point at the same checkpoint
file.
"""
from __future__ import annotations

import json

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from orchestration import config
from orchestration.db.connection import get_connection
from orchestration.hitl import escalation
from orchestration.hitl.approval_queue import create as create_approval
from orchestration.hitl.approval_queue import resolve as resolve_approval
from orchestration.memory import long_term_memory, working_memory
from orchestration.reviewer import review_subtask_output
from orchestration.schemas import EscalationLevel, ExecutionPlan, ReviewResult, Subtask
from orchestration.specialists.base import SpecialistFailure, run_specialist
from orchestration.state import GraphState
from orchestration.supervisor import create_plan
from orchestration.synthesis import synthesize
from orchestration.tracing.tracer import new_id, start_span

MAX_TOOL_RETRY_HINT = "Reviewer feedback from a previous attempt on this subtask:"


# ---------------------------------------------------------------- helpers

def _plan(state: GraphState) -> ExecutionPlan:
    return ExecutionPlan.model_validate(state["plan"])


def _subtask_by_id(plan: ExecutionPlan, subtask_id: str) -> Subtask:
    for st in plan.subtasks:
        if st.id == subtask_id:
            return st
    raise KeyError(subtask_id)


def _next_runnable_subtask(plan: ExecutionPlan, outputs: dict[str, str]) -> Subtask | None:
    for st in plan.subtasks:
        if st.id in outputs:
            continue
        if all(dep in outputs for dep in st.depends_on):
            return st
    return None


# -------------------------------------------------------------------- nodes

def intake_node(state: GraphState) -> dict:
    conn = get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO tasks (id, user_id, request, status) VALUES (?, ?, ?, 'planning')",
        (state["task_id"], state["user_id"], state["request"]),
    )
    conn.commit()

    try:
        hits = long_term_memory.recall(user_id=state["user_id"], query=state["request"], n_results=5)
        memories = [{"text": h.text, "kind": h.kind, "importance": h.importance} for h in hits]
    except Exception as exc:  # noqa: BLE001 - memory is an enhancement, not a hard dependency: a
        # transient embedding/Chroma failure shouldn't take down the whole task before it even has a
        # plan. Recorded as a failed span (not silently swallowed) so it's visible in Trace Explorer.
        memories = []
        span = start_span(task_id=state["task_id"], agent="supervisor", span_type="plan", name="memory_recall_degraded")
        span.finish(status="failure", error=str(exc))

    return {"memories": memories}


def plan_node(state: GraphState) -> dict:
    from orchestration.memory.long_term_memory import MemoryHit

    memory_hits = [
        MemoryHit(id="", text=m["text"], kind=m["kind"], user_id=state["user_id"], metadata={}, distance=0.0,
                   importance=m["importance"])
        for m in state.get("memories", [])
    ]
    plan = create_plan(task_id=state["task_id"], request=state["request"], memories=memory_hits)

    conn = get_connection()
    conn.execute(
        "INSERT INTO execution_plans (id, task_id, subtasks_json, confidence, reasoning) VALUES (?, ?, ?, ?, ?)",
        (new_id("plan"), state["task_id"], json.dumps([s.model_dump() for s in plan.subtasks]),
         plan.confidence, plan.reasoning),
    )
    conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (state["task_id"],))
    conn.commit()

    decision = escalation.check_plan(plan)
    return {
        "plan": plan.model_dump(),
        "subtask_outputs": {},
        "subtask_retry_counts": {},
        "review_retry_counts": {},
        "approved_sensitive_subtasks": [],
        "needs_plan_approval": decision is not None,
        "status": "awaiting_approval" if decision else "running",
    }


def route_after_plan(state: GraphState) -> str:
    return "raise_plan_escalation" if state.get("needs_plan_approval") else "select_subtask"


def raise_plan_escalation_node(state: GraphState) -> dict:
    """Creates the approval request and packages the interrupt payload.

    Split out from await_plan_decision_node deliberately: LangGraph re-runs
    a node's body from the top whenever it resumes past an interrupt() call
    inside that same node, so anything with a side effect (like inserting
    an approval_requests row) has to live in a node that completes -- and
    therefore checkpoints -- *before* the interrupting node, or it fires
    twice (once on the initial pause, once again on resume).
    """
    plan = _plan(state)
    decision = escalation.check_plan(plan) or escalation.EscalationDecision(EscalationLevel.APPROVE_PLAN, "manual review requested")

    request = create_approval(
        task_id=state["task_id"], level=decision.level, trigger_reason=decision.reason,
        context={"request": state["request"], "plan": state["plan"], "memories": state.get("memories", [])},
        proposed_action={"type": "execute_plan", "plan": state["plan"]},
    )
    return {
        "pending_approval_id": request.id,
        "pending_approval_payload": {
            "approval_id": request.id, "kind": "plan", "level": decision.level, "reason": decision.reason,
            "plan": state["plan"],
        },
    }


def await_plan_decision_node(state: GraphState) -> dict:
    resume_value = interrupt(state["pending_approval_payload"])

    decision_kind = resume_value.get("decision", "reject")
    resolve_approval(
        state["pending_approval_id"],
        status={"approve": "approved", "modify": "modified", "take_over": "take_over"}.get(decision_kind, "rejected"),
        resolution=resume_value, reviewer_notes=resume_value.get("notes"),
    )

    if decision_kind == "approve":
        return {"plan_status": "approved", "status": "running"}
    if decision_kind == "modify":
        new_plan = ExecutionPlan.model_validate(resume_value["plan"])
        return {"plan": new_plan.model_dump(), "plan_status": "approved", "status": "running"}
    if decision_kind == "take_over":
        return {"plan_status": "took_over", "final_output": resume_value.get("output", ""), "status": "completed"}
    return {"plan_status": "rejected", "final_output": f"Task rejected by reviewer: {resume_value.get('notes', '')}",
            "status": "failed"}


def route_after_plan_decision(state: GraphState) -> str:
    status = state.get("plan_status")
    if status == "approved":
        return "select_subtask"
    return "delivery"  # rejected or took_over -> skip straight to wrap-up


def select_subtask_node(state: GraphState) -> dict:
    plan = _plan(state)
    outputs = state.get("subtask_outputs", {})
    nxt = _next_runnable_subtask(plan, outputs)
    if nxt is None:
        # No runnable subtask does NOT necessarily mean "all done" -- it
        # also happens if the remaining subtasks are permanently blocked
        # (a dangling depends_on, a cycle). ExecutionPlan's validator
        # rejects those at parse time, but this check is the last line of
        # defense against ever silently synthesizing a "completed" answer
        # over a gap, e.g. if validation is ever bypassed.
        unresolved = [st.id for st in plan.subtasks if st.id not in outputs]
        return {"current_subtask_id": None, "plan_blocked": bool(unresolved), "blocked_subtask_ids": unresolved}

    already_approved = nxt.id in state.get("approved_sensitive_subtasks", [])
    needs_approval = (not already_approved) and escalation.check_subtask_text(nxt.description, nxt.expected_output_format) is not None
    return {"current_subtask_id": nxt.id, "subtask_needs_approval": needs_approval, "plan_blocked": False}


def route_after_select(state: GraphState) -> str:
    if state.get("plan_blocked"):
        return "plan_blocked"
    if state.get("current_subtask_id") is None:
        return "synthesis"
    if state.get("subtask_needs_approval"):
        return "raise_action_escalation"
    return "execute_subtask"


def plan_blocked_node(state: GraphState) -> dict:
    ids = state.get("blocked_subtask_ids", [])
    return {
        "final_output": (
            f"Task failed: subtask(s) {ids} never became runnable (their dependencies never "
            "completed). This indicates a plan with a dependency the system couldn't satisfy."
        ),
        "status": "failed",
    }


def raise_action_escalation_node(state: GraphState) -> dict:
    """See raise_plan_escalation_node's docstring for why this is split
    from await_action_decision_node."""
    plan = _plan(state)
    subtask_id = state["current_subtask_id"]
    subtask = _subtask_by_id(plan, subtask_id)

    retry_count = state.get("subtask_retry_counts", {}).get(subtask_id, 0)
    review_retries = state.get("review_retry_counts", {}).get(subtask_id, 0)
    trigger = (
        escalation.check_subtask_text(subtask.description, subtask.expected_output_format)
        or escalation.check_specialist_failure(retry_count)
        or (escalation.check_review(ReviewResult.model_validate(state["last_review"])) if state.get("last_review") else None)
        or escalation.EscalationDecision(EscalationLevel.APPROVE_ACTION, "manual review requested")
    )

    request = create_approval(
        task_id=state["task_id"], level=trigger.level, trigger_reason=trigger.reason,
        context={
            "subtask": subtask.model_dump(), "retry_count": retry_count, "review_retries": review_retries,
            "last_error": state.get("last_error"), "last_review": state.get("last_review"),
            "attempted_output": state.get("subtask_outputs", {}).get(subtask_id),
        },
        proposed_action={"type": "run_subtask", "subtask": subtask.model_dump()},
    )
    return {
        "pending_approval_id": request.id,
        "pending_approval_payload": {
            "approval_id": request.id, "kind": "action", "level": trigger.level, "reason": trigger.reason,
            "subtask": subtask.model_dump(),
        },
    }


def await_action_decision_node(state: GraphState) -> dict:
    subtask_id = state["current_subtask_id"]
    resume_value = interrupt(state["pending_approval_payload"])

    decision_kind = resume_value.get("decision", "reject")
    resolve_approval(
        state["pending_approval_id"],
        status={"approve": "approved", "modify": "modified", "take_over": "take_over"}.get(decision_kind, "rejected"),
        resolution=resume_value, reviewer_notes=resume_value.get("notes"),
    )
    outputs = dict(state.get("subtask_outputs", {}))
    approved_sensitive = list(state.get("approved_sensitive_subtasks", []))

    if decision_kind == "approve":
        approved_sensitive.append(subtask_id)
        return {"approved_sensitive_subtasks": approved_sensitive, "resolved_action": "approved"}
    if decision_kind == "modify":
        outputs[subtask_id] = resume_value.get("output", "")
        return {"subtask_outputs": outputs, "resolved_action": "modified"}
    if decision_kind == "take_over":
        outputs[subtask_id] = resume_value.get("output", "")
        return {"subtask_outputs": outputs, "resolved_action": "took_over"}

    # Rejecting a subtask must not silently continue: it used to stuff a
    # "[SKIPPED...]" placeholder string into subtask_outputs and carry on to
    # synthesis/select_subtask as if that string were real specialist
    # output -- which any downstream dependent subtask would then receive
    # as if it were legitimate context, and the task would still be marked
    # "completed". A rejection means the human declined to let this step
    # happen; that has to end the task, the same way rejecting the plan
    # itself does.
    return {
        "resolved_action": "rejected",
        "status": "failed",
        "final_output": f"Task failed: subtask '{subtask_id}' rejected by human reviewer -- {resume_value.get('notes', '')}",
    }


def route_after_action_decision(state: GraphState) -> str:
    if state.get("resolved_action") == "rejected":
        return "delivery"
    return "select_subtask"


def execute_subtask_node(state: GraphState) -> dict:
    plan = _plan(state)
    subtask = _subtask_by_id(plan, state["current_subtask_id"])
    context_inputs = {dep: state["subtask_outputs"][dep] for dep in subtask.depends_on if dep in state["subtask_outputs"]}

    review_retries = state.get("review_retry_counts", {}).get(subtask.id, 0)
    if review_retries > 0 and state.get("last_review"):
        context_inputs["reviewer_feedback"] = f"{MAX_TOOL_RETRY_HINT} {state['last_review'].get('feedback', '')}"

    # A specialist-failure retry (as opposed to a review-triggered retry, handled above) previously
    # got no information about what went wrong last time -- it just blindly reran the same prompt and
    # often failed the same way again. Surface the working-memory error log for this subtask so the
    # retry can actually adapt. This is also what makes working_memory's error log something read
    # back, not just written and never consulted.
    if state.get("subtask_retry_counts", {}).get(subtask.id, 0) > 0:
        past_errors = [e for e in working_memory.get_value(state["task_id"], "errors", default=[]) if e.get("subtask") == subtask.id]
        if past_errors:
            context_inputs["previous_attempt_errors"] = (
                "Prior attempt(s) at this subtask failed with: "
                + "; ".join(f"[{e.get('kind', 'error')}] {e.get('error', '')}" for e in past_errors)
            )

    try:
        output = run_specialist(task_id=state["task_id"], subtask=subtask, context_inputs=context_inputs)
    except SpecialistFailure as exc:
        error_kind, error_message = "specialist_did_not_converge", str(exc)
    except Exception as exc:  # noqa: BLE001 - graceful degradation for genuinely unexpected errors too
        # (a bug in our own code, not just an LLM/tool failure), but tagged with a distinct `kind` so
        # it doesn't read identically to an expected specialist failure in the trace/error log.
        error_kind, error_message = "unexpected_error", str(exc)
    else:
        outputs = dict(state.get("subtask_outputs", {}))
        outputs[subtask.id] = output
        return {"subtask_outputs": outputs, "last_subtask_status": "success", "last_error": None}

    retries = dict(state.get("subtask_retry_counts", {}))
    retries[subtask.id] = retries.get(subtask.id, 0) + 1
    working_memory.append_log(state["task_id"], "errors", {"subtask": subtask.id, "kind": error_kind, "error": error_message})
    return {"subtask_retry_counts": retries, "last_subtask_status": "failed", "last_error": error_message}


def route_after_execute(state: GraphState) -> str:
    if state.get("last_subtask_status") == "failed":
        subtask_id = state["current_subtask_id"]
        retry_count = state.get("subtask_retry_counts", {}).get(subtask_id, 0)
        if escalation.check_specialist_failure(retry_count):
            return "raise_action_escalation"
        return "select_subtask"  # will re-select the same (still-missing) subtask -> natural retry
    return "review_subtask"


def review_subtask_node(state: GraphState) -> dict:
    plan = _plan(state)
    subtask = _subtask_by_id(plan, state["current_subtask_id"])
    output = state["subtask_outputs"][subtask.id]
    review = review_subtask_output(task_id=state["task_id"], subtask=subtask, output=output)

    if review.passed:
        return {"last_review": review.model_dump(), "review_status": "passed"}

    review_retries = dict(state.get("review_retry_counts", {}))
    review_retries[subtask.id] = review_retries.get(subtask.id, 0) + 1

    # The retry cap is the sole gate, deliberately NOT gated on
    # escalation.check_review() (which only looks at quality_score). The
    # reviewer sets `passed` and `quality_score` independently -- nothing
    # guarantees they agree, and a `passed=False` review with a
    # threshold-clearing score used to mean check_review() returned None
    # forever, so this branch never fired and the subtask retried
    # indefinitely (execute succeeds -> review rejects -> delete output ->
    # retry -> forever), since subtask_retry_counts' cap only triggers on
    # specialist *exceptions*, not on review rejection. Using the same
    # settings knob as the specialist-failure cap in route_after_execute
    # keeps both retry limits consistent and both bounded.
    if review_retries[subtask.id] >= config.settings.max_specialist_retries:
        return {"last_review": review.model_dump(), "review_retry_counts": review_retries, "review_status": "needs_escalation"}

    outputs = dict(state["subtask_outputs"])
    del outputs[subtask.id]  # force a re-run with feedback injected
    return {"subtask_outputs": outputs, "last_review": review.model_dump(),
            "review_retry_counts": review_retries, "review_status": "needs_retry"}


def route_after_review(state: GraphState) -> str:
    status = state.get("review_status")
    if status == "needs_escalation":
        return "raise_action_escalation"
    return "select_subtask"  # covers both "passed" (moves to next) and "needs_retry" (re-selects same one)


def synthesis_node(state: GraphState) -> dict:
    plan = _plan(state)
    final = synthesize(task_id=state["task_id"], request=state["request"], plan=plan, subtask_outputs=state["subtask_outputs"])
    return {"final_output": final, "status": "completed"}


def delivery_node(state: GraphState) -> dict:
    conn = get_connection()
    conn.execute(
        "UPDATE tasks SET status = ?, result = ?, completed_at = datetime('now') WHERE id = ?",
        (state.get("status", "completed"), state.get("final_output", ""), state["task_id"]),
    )
    conn.commit()

    if state.get("status") == "completed":
        long_term_memory.remember(
            user_id=state["user_id"], kind="task_summary",
            text=f"Request: {state['request']}\nOutcome: {state['final_output']}",
            metadata={"task_id": state["task_id"]},
        )

    working_memory.clear(state["task_id"])
    return {}


# ------------------------------------------------------------------- build

def build_graph():
    g = StateGraph(GraphState)
    g.add_node("intake", intake_node)
    g.add_node("plan", plan_node)
    g.add_node("raise_plan_escalation", raise_plan_escalation_node)
    g.add_node("await_plan_decision", await_plan_decision_node)
    g.add_node("select_subtask", select_subtask_node)
    g.add_node("plan_blocked", plan_blocked_node)
    g.add_node("raise_action_escalation", raise_action_escalation_node)
    g.add_node("await_action_decision", await_action_decision_node)
    g.add_node("execute_subtask", execute_subtask_node)
    g.add_node("review_subtask", review_subtask_node)
    g.add_node("synthesis", synthesis_node)
    g.add_node("delivery", delivery_node)

    g.add_edge(START, "intake")
    g.add_edge("intake", "plan")
    g.add_conditional_edges("plan", route_after_plan, {"raise_plan_escalation": "raise_plan_escalation", "select_subtask": "select_subtask"})
    g.add_edge("raise_plan_escalation", "await_plan_decision")
    g.add_conditional_edges("await_plan_decision", route_after_plan_decision, {"select_subtask": "select_subtask", "delivery": "delivery"})
    g.add_conditional_edges("select_subtask", route_after_select,
                             {"synthesis": "synthesis", "plan_blocked": "plan_blocked",
                              "raise_action_escalation": "raise_action_escalation", "execute_subtask": "execute_subtask"})
    g.add_edge("plan_blocked", "delivery")
    g.add_edge("raise_action_escalation", "await_action_decision")
    g.add_conditional_edges("await_action_decision", route_after_action_decision,
                             {"select_subtask": "select_subtask", "delivery": "delivery"})
    g.add_conditional_edges("execute_subtask", route_after_execute,
                             {"raise_action_escalation": "raise_action_escalation", "select_subtask": "select_subtask", "review_subtask": "review_subtask"})
    g.add_conditional_edges("review_subtask", route_after_review, {"raise_action_escalation": "raise_action_escalation", "select_subtask": "select_subtask"})
    g.add_edge("synthesis", "delivery")
    g.add_edge("delivery", END)
    return g
