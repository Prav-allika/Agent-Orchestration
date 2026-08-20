#!/usr/bin/env python
"""End-to-end showcase scenario (spec Phase 5.1):

A research task that needs web search, data extraction/analysis, and a
written summary -- with a step deliberately worded to trip the sensitive-
operation escalation trigger (mentions emailing the result externally), so
the demo exercises supervisor planning, parallel-ish specialist work,
review, human approval, and memory in one run.

Run interactively:      python demo/run_demo.py
Run unattended (CI):    python demo/run_demo.py --auto-approve
Run the memory follow-up: python demo/run_demo.py --memory-followup
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestration.graph_runner import resume_task, start_task  # noqa: E402
from orchestration.memory.long_term_memory import dashboard_snapshot  # noqa: E402
from orchestration.tracing.tracer import get_task_cost_summary  # noqa: E402

DEFAULT_SCENARIO = (
    "Research current adoption trends for small modular nuclear reactors (SMRs) in 2025-2026, "
    "calculate the approximate percent growth in announced projects compared to 2023, and write a "
    "250-300 word executive summary with cited sources. Then send the final summary to the "
    "compliance team via email for external distribution."
)

FOLLOWUP_SCENARIO = (
    "Research current adoption trends for offshore wind energy in 2025-2026 and write a 250-word "
    "executive summary with cited sources, in the same style as the SMR summary you wrote before."
)


def prompt_decision(interrupt_payload: dict) -> dict:
    print("\n" + "=" * 70)
    print(f"  ESCALATION: {interrupt_payload['kind']} / level={interrupt_payload['level']}")
    print(f"  Reason: {interrupt_payload['reason']}")
    print("=" * 70)
    if interrupt_payload["kind"] == "plan":
        for st in interrupt_payload["plan"]["subtasks"]:
            print(f"  - [{st['specialist']}] {st['id']}: {st['description']}")
        print(f"  confidence={interrupt_payload['plan']['confidence']:.2f} "
              f"sensitive={interrupt_payload['plan']['is_sensitive']}")
    else:
        st = interrupt_payload["subtask"]
        print(f"  Subtask: [{st['specialist']}] {st['description']}")

    choice = input("\nApprove / Reject / notes> Type 'a' to approve, 'r' to reject: ").strip().lower()
    if choice == "r":
        notes = input("Rejection notes: ")
        return {"decision": "reject", "notes": notes}
    return {"decision": "approve"}


def run_scenario(request: str, *, user_id: str, auto_approve: bool) -> str | None:
    print(f"\n>>> Submitting task for user '{user_id}':\n{request}\n")
    result = start_task(user_id=user_id, request=request)

    while result.status == "awaiting_approval":
        if auto_approve:
            print(f"\n[auto-approve] resolving escalation: {result.interrupt['reason']}")
            decision = {"decision": "approve"}
        else:
            decision = prompt_decision(result.interrupt)
        result = resume_task(task_id=result.task_id, resume_value=decision)

    print(f"\n>>> Task {result.task_id} finished with status: {result.status}\n")
    if result.final_output:
        print(result.final_output)

    cost = get_task_cost_summary(result.task_id)
    print(f"\n--- Cost/perf summary: {cost.get('span_count', 0)} spans, "
          f"${cost.get('cost_usd') or 0:.4f}, {cost.get('latency_ms_total') or 0:.0f}ms total ---")
    return result.task_id


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user-id", default="demo_user")
    parser.add_argument("--auto-approve", action="store_true", help="Resolve every escalation as 'approve' automatically")
    parser.add_argument("--memory-followup", action="store_true", help="Also run a related follow-up task to show memory informing planning")
    args = parser.parse_args()

    run_scenario(DEFAULT_SCENARIO, user_id=args.user_id, auto_approve=args.auto_approve)

    if args.memory_followup:
        print("\n\n########## MEMORY-INFORMED FOLLOW-UP ##########")
        run_scenario(FOLLOWUP_SCENARIO, user_id=args.user_id, auto_approve=args.auto_approve)

        print("\n--- Long-term memory dashboard for this user ---")
        for mem in dashboard_snapshot(args.user_id):
            print(f"  [{mem['kind']}, importance={mem['importance']}] {mem['text'][:120]}...")


if __name__ == "__main__":
    main()
