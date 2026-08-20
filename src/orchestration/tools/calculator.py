"""Arithmetic evaluator. A dedicated tool rather than routing simple math
through code_exec keeps trivial calculations cheap, fast, and outside the
subprocess sandbox entirely.
"""
from __future__ import annotations

import ast
import operator

from pydantic import BaseModel, Field

from orchestration.tools.registry import ToolRegistry

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


class CalculatorInput(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. '(3 + 4) * 2 / 7'")


class CalculatorOutput(BaseModel):
    result: float


def run(input_data: CalculatorInput) -> CalculatorOutput:
    try:
        tree = ast.parse(input_data.expression, mode="eval")
        return CalculatorOutput(result=_eval(tree.body))
    except Exception as exc:
        raise ValueError(f"Could not evaluate '{input_data.expression}': {exc}") from exc


def register(registry: ToolRegistry) -> None:
    registry.register(
        name="calculator",
        description="Evaluate a pure arithmetic expression (+ - * / % **). No variables or function calls.",
        input_schema=CalculatorInput,
        output_schema=CalculatorOutput,
        allowed_agents=["data_analysis", "research", "writer"],
        rate_limit_per_minute=60,
        func=run,
    )
