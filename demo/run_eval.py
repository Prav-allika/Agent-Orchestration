#!/usr/bin/env python
"""Run the golden-dataset eval suite: executes every task in
orchestration.eval.golden_tasks through the real graph (auto-approving
escalations), scores each result with an LLM judge against its rubric, and
prints a pass/fail summary. Every result is persisted to eval_results for
run-over-run comparison (see orchestration.eval.runner.get_eval_run_history).

Usage: python demo/run_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestration.eval.runner import run_eval_suite  # noqa: E402


def main():
    print(">>> Running golden-dataset eval suite...\n")
    summary = run_eval_suite()

    for r in summary["results"]:
        icon = "✅" if r.judge_passed else ("❌" if r.judge_passed is False else "⚠️ ")
        print(f"{icon} {r.golden_task_id:35s} task_status={r.task_status:12s} judge_score={r.judge_score}")

    print(f"\n>>> Eval run {summary['eval_run_id']}: {summary['passed']}/{summary['judged']} judged tasks passed"
          f" (rate={summary['pass_rate']:.0%})" if summary["pass_rate"] is not None else "\n>>> No tasks were judged.")


if __name__ == "__main__":
    main()
