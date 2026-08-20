"""Web search tool, backed by DuckDuckGo's HTML endpoint.

No API key required (unlike Bing/Google/Serp APIs), which matters for this
MVP since only an OpenAI key is available. Swappable later for a paid
search API by rewriting only `_search`.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from orchestration.tools.registry import ToolRegistry


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, ge=1, le=10)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class WebSearchOutput(BaseModel):
    results: list[SearchResult]


def _search(query: str, max_results: int) -> list[SearchResult]:
    import requests
    from bs4 import BeautifulSoup

    resp = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": "Mozilla/5.0 (agent-orchestration-mvp)"},
        timeout=10,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results: list[SearchResult] = []
    for result_div in soup.select("div.result"):
        link = result_div.select_one("a.result__a")
        snippet_el = result_div.select_one("a.result__snippet") or result_div.select_one(".result__snippet")
        if not link:
            continue
        results.append(
            SearchResult(
                title=link.get_text(strip=True),
                url=link.get("href", ""),
                snippet=snippet_el.get_text(strip=True) if snippet_el else "",
            )
        )
        if len(results) >= max_results:
            break
    return results


def run(input_data: WebSearchInput) -> WebSearchOutput:
    try:
        results = _search(input_data.query, input_data.max_results)
    except Exception as exc:
        raise RuntimeError(f"web_search failed: {exc}") from exc
    return WebSearchOutput(results=results)


def register(registry: ToolRegistry) -> None:
    registry.register(
        name="web_search",
        description="Search the web via DuckDuckGo and return titles, URLs, and snippets.",
        input_schema=WebSearchInput,
        output_schema=WebSearchOutput,
        allowed_agents=["research"],
        rate_limit_per_minute=20,
        func=run,
    )
