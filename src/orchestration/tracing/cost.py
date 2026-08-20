"""Token pricing table and cost computation, shared by the LLM provider
(to price each call) and the trace aggregation queries (to roll costs up).
"""
from __future__ import annotations

# USD per 1M tokens, (prompt, completion). Update as pricing changes; this
# is intentionally a static table rather than a live-fetched one so cost
# figures are reproducible across replays.
PRICING_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-3.5-turbo": (0.50, 1.50),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-sol": (5.00, 30.00),
}

DEFAULT_PRICE = (0.50, 1.50)


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prompt_price, completion_price = PRICING_PER_MILLION.get(model, DEFAULT_PRICE)
    return (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1_000_000
