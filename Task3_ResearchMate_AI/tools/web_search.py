"""Real-time web search via Tavily (if a key is set) or DuckDuckGo (no key needed)."""
from __future__ import annotations

from typing import Callable
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from config.settings import Settings
from core.exceptions import SearchError, ToolInputError
from core.logger import get_logger, redact
from core.models import Source, ToolResult
from core.validators import validate_text

log = get_logger(__name__)

# A backend takes (query, max_results, timeout_seconds) and returns dicts with
# the keys "title", "url" and "snippet".
SearchBackend = Callable[[str, int, float], list[dict]]

DISCLAIMER = (
    "NOTE: These are unverified snippets retrieved from third-party websites. They may be "
    "inaccurate, outdated or biased. Treat them as retrieved information, not verified facts."
)


class WebSearchInput(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=300,
        description="Natural-language search query, e.g. 'current population of Japan 2026'.",
    )
    max_results: int | None = Field(
        default=None, ge=1, le=10, description="Optional number of results (default 5)."
    )


# ----------------------------------------------------------------- backends
def duckduckgo_backend(query: str, max_results: int, timeout: float) -> list[dict]:
    from ddgs import DDGS  # imported lazily so tests can run without network

    with DDGS(timeout=int(timeout)) as client:
        raw = client.text(query, max_results=max_results)
    return [
        {"title": r.get("title", ""), "url": r.get("href") or r.get("url", ""), "snippet": r.get("body", "")}
        for r in raw or []
    ]


def tavily_backend_factory(api_key: str) -> SearchBackend:
    def backend(query: str, max_results: int, timeout: float) -> list[dict]:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results, timeout=timeout)
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in response.get("results", [])
        ]

    return backend


def _classify(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if isinstance(exc, TimeoutError) or "timeout" in name or "timed out" in text:
        return "the request timed out"
    if "ratelimit" in name or "rate limit" in text or "429" in text:
        return "the search provider is rate limiting requests"
    if isinstance(exc, (ConnectionError, OSError)) or "connect" in name or "network" in text:
        return "a network error occurred"
    return f"the provider returned an error ({type(exc).__name__})"


def _clean(results: list[dict], limit: int) -> list[dict]:
    seen: set[str] = set()
    cleaned: list[dict] = []
    for r in results:
        url = str(r.get("url") or "").strip()
        if urlparse(url).scheme not in ("http", "https") or url in seen:
            continue
        seen.add(url)
        cleaned.append({
            "title": (str(r.get("title") or "").strip() or url)[:150],
            "url": url,
            "snippet": " ".join(str(r.get("snippet") or "").split())[:500],
        })
        if len(cleaned) >= limit:
            break
    return cleaned


class WebSearcher:
    def __init__(self, settings: Settings, backends: dict[str, SearchBackend] | None = None):
        self.settings = settings
        self.backends: dict[str, SearchBackend] = {"duckduckgo": duckduckgo_backend}
        if settings.tavily_api_key:
            self.backends["tavily"] = tavily_backend_factory(settings.tavily_api_key)
        if backends:
            self.backends.update(backends)

    def provider_order(self) -> list[str]:
        choice = self.settings.search_provider
        if choice == "duckduckgo":
            return ["duckduckgo"]
        if choice == "tavily":
            return ["tavily"]
        return ["tavily", "duckduckgo"] if "tavily" in self.backends else ["duckduckgo"]

    def search(self, query: str, max_results: int | None = None) -> ToolResult:
        try:
            return self._search(query, max_results)
        except (ToolInputError, SearchError) as exc:
            return ToolResult.failure("web_search", redact(exc))

    def _search(self, query: str, max_results: int | None) -> ToolResult:
        query = validate_text(query, name="query", min_len=2, max_len=300)
        limit = min(max_results or self.settings.max_search_results, self.settings.max_search_results)
        limit = max(limit, 1)

        errors: list[str] = []
        for provider in self.provider_order():
            backend = self.backends.get(provider)
            if backend is None:
                errors.append(f"{provider}: not configured (missing API key)")
                continue
            try:
                raw = backend(query, limit, float(self.settings.search_timeout))
            except Exception as exc:  # provider libraries raise many exception types
                reason = _classify(exc)
                if "no results" in str(exc).lower():
                    continue
                log.info("Search provider %s failed: %s", provider, exc)
                errors.append(f"{provider}: {reason}")
                continue
            results = _clean(raw, limit)
            if results:
                return self._format(query, provider, results)

        if errors:
            raise SearchError("Web search failed - " + "; ".join(errors) + ". Try again later or rephrase.")
        return ToolResult.success(
            "web_search",
            content=f'No web results found for: "{query}". Try different or broader keywords.',
            summary="No results found",
        )

    @staticmethod
    def _format(query: str, provider: str, results: list[dict]) -> ToolResult:
        lines = [f'Web search results for: "{query}" (provider: {provider}, {len(results)} results)', DISCLAIMER, ""]
        for i, r in enumerate(results, 1):
            lines += [f"[{i}] {r['title']}", f"URL: {r['url']}", f"Snippet: {r['snippet']}", ""]
        return ToolResult.success(
            "web_search",
            content="\n".join(lines).rstrip(),
            summary=f"Retrieved {len(results)} results via {provider} (unverified snippets)",
            sources=[Source(title=r["title"], url=r["url"], kind="web") for r in results],
            untrusted=True,
        )
