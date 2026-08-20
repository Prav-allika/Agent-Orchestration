"""Shared structured-output schemas used across the supervisor, specialists,
reviewer, and graph state. Centralized here so every module imports the
same definitions instead of redefining compatible-but-not-identical copies.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

SpecialistName = Literal["research", "data_analysis", "writer", "code_exec"]
Complexity = Literal["low", "medium", "high"]


class Subtask(BaseModel):
    id: str = Field(description="Short unique id, e.g. 'st1'")
    description: str
    specialist: SpecialistName
    depends_on: list[str] = Field(default_factory=list)
    required_inputs: str = Field(description="What this subtask needs, drawn from prior subtask outputs or the user request")
    expected_output_format: str
    complexity: Complexity


class ExecutionPlan(BaseModel):
    subtasks: list[Subtask] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1, description="Supervisor's confidence this plan will satisfy the request")
    reasoning: str
    is_sensitive: bool = Field(description="True if any subtask involves financial transactions, data deletion, "
                                            "or external communications (email/posting/messaging)")

    @model_validator(mode="after")
    def _validate_dependency_graph(self) -> "ExecutionPlan":
        """Catches structurally broken plans (a hallucinated depends_on that
        references a nonexistent subtask, a dependency cycle) before they
        ever reach the graph. Without this, _next_runnable_subtask() in
        graph.py would silently never select the blocked subtask(s), and
        the task would complete with a gap papered over as "(missing)" in
        the final synthesis -- wrong output reported as success. Runs on
        every ExecutionPlan.model_validate() call, including the "modify
        plan" path in the Streamlit approval UI, not just supervisor output.
        """
        ids = [s.id for s in self.subtasks]
        id_set = set(ids)
        if len(id_set) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate subtask ids: {dupes}")

        dangling = [(s.id, dep) for s in self.subtasks for dep in s.depends_on if dep not in id_set]
        if dangling:
            raise ValueError(f"subtask(s) depend on nonexistent subtask ids: {dangling}")

        # DFS cycle detection over the depends_on graph.
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {s.id: WHITE for s in self.subtasks}
        deps = {s.id: s.depends_on for s in self.subtasks}

        def visit(node: str, path: list[str]) -> None:
            color[node] = GRAY
            for dep in deps[node]:
                if color[dep] == GRAY:
                    cycle = path[path.index(dep):] + [dep]
                    raise ValueError(f"circular dependency among subtasks: {' -> '.join(cycle)}")
                if color[dep] == WHITE:
                    visit(dep, path + [dep])
            color[node] = BLACK

        for s in self.subtasks:
            if color[s.id] == WHITE:
                visit(s.id, [s.id])

        return self


class ReviewResult(BaseModel):
    passed: bool
    quality_score: float = Field(ge=0, le=1)
    feedback: str
    issues: list[str] = Field(default_factory=list)


class EscalationLevel:
    NOTIFY = "notify"
    APPROVE_ACTION = "approve_action"
    APPROVE_PLAN = "approve_plan"
    TAKE_OVER = "take_over"
