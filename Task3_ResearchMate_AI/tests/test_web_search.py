import sys
import types
from dataclasses import replace

import pytest
from pydantic import ValidationError

from tests.conftest import FAKE_TAVILY_KEY
from tools.web_search import WebSearcher, WebSearchInput, duckduckgo_backend


def make(settings, backend, **overrides):
    cfg = replace(settings, **overrides) if overrides else settings
    return WebSearcher(cfg, backends={"duckduckgo": backend})


def results(n):
    return [{"title": f"Title {i}", "url": f"https://example.com/{i}", "snippet": f"Snippet {i}"} for i in range(n)]


def test_successful_search_includes_titles_urls_and_disclaimer(settings):
    r = make(settings, lambda q, n, t: results(3)).search("latest AI news")
    assert r.ok and r.untrusted
    assert "Title 0" in r.content and "https://example.com/0" in r.content
    assert "unverified" in r.content.lower()
    assert [s.url for s in r.sources] == [f"https://example.com/{i}" for i in range(3)]


def test_result_count_is_limited(settings):
    r = make(settings, lambda q, n, t: results(10), max_search_results=3).search("anything")
    assert len(r.sources) == 3
    r2 = make(settings, lambda q, n, t: results(10), max_search_results=5).search("anything", max_results=2)
    assert len(r2.sources) == 2


def test_backend_receives_limit_and_timeout(settings):
    seen = {}

    def backend(q, n, t):
        seen.update(q=q, n=n, t=t)
        return results(1)

    make(settings, backend, max_search_results=4, search_timeout=9).search("query text")
    assert seen == {"q": "query text", "n": 4, "t": 9.0}


def test_duplicates_and_unsafe_urls_removed(settings):
    raw = [
        {"title": "A", "url": "https://a.com", "snippet": ""},
        {"title": "A again", "url": "https://a.com", "snippet": ""},
        {"title": "JS", "url": "javascript:alert(1)", "snippet": ""},
        {"title": "File", "url": "file:///etc/passwd", "snippet": ""},
        {"title": "B", "url": "http://b.com", "snippet": ""},
    ]
    r = make(settings, lambda q, n, t: raw).search("dedupe me")
    assert [s.url for s in r.sources] == ["https://a.com", "http://b.com"]


def test_empty_results_are_not_an_error(settings):
    r = make(settings, lambda q, n, t: []).search("zzzxxqqq nothing")
    assert r.ok and "No web results found" in r.content and not r.sources


def test_ddgs_no_results_exception_is_treated_as_empty(settings):
    def backend(q, n, t):
        raise RuntimeError("No results found.")

    r = make(settings, backend).search("rare query")
    assert r.ok and "No web results" in r.content


def test_network_error_returns_readable_failure(settings):
    def backend(q, n, t):
        raise ConnectionError("dns failure")

    r = make(settings, backend).search("anything at all")
    assert not r.ok and "network error" in r.error


def test_timeout_returns_readable_failure(settings):
    def backend(q, n, t):
        raise TimeoutError("read timed out")

    r = make(settings, backend).search("anything at all")
    assert not r.ok and "timed out" in r.error


def test_rate_limit_failure_is_classified(settings):
    class RatelimitException(Exception):
        pass

    def backend(q, n, t):
        raise RatelimitException("202 Ratelimit")

    r = make(settings, backend).search("anything at all")
    assert not r.ok and "rate limiting" in r.error


@pytest.mark.parametrize("query", ["", " ", "a", "x" * 301, "bad\x00query", None, 123])
def test_invalid_queries_rejected_by_searcher(settings, query):
    r = make(settings, lambda q, n, t: results(1)).search(query)  # type: ignore[arg-type]
    assert not r.ok


@pytest.mark.parametrize("payload", [
    {"query": ""}, {"query": "x" * 301}, {"query": "ok query", "max_results": 0},
    {"query": "ok query", "max_results": 99}, {},
])
def test_input_schema_rejects_bad_arguments(payload):
    with pytest.raises(ValidationError):
        WebSearchInput(**payload)


def test_tavily_preferred_when_key_present_with_ddg_fallback(settings):
    calls = []

    def tavily(q, n, t):
        calls.append("tavily")
        raise ConnectionError("down")

    def ddg(q, n, t):
        calls.append("ddg")
        return results(1)

    cfg = replace(settings, tavily_api_key="tvly-test-key-123456")
    searcher = WebSearcher(cfg, backends={"tavily": tavily, "duckduckgo": ddg})
    assert searcher.provider_order() == ["tavily", "duckduckgo"]
    r = searcher.search("fallback please")
    assert r.ok and calls == ["tavily", "ddg"]
    assert "provider: duckduckgo" in r.content


def test_explicit_tavily_without_key_fails_clearly(settings):
    searcher = WebSearcher(replace(settings, search_provider="tavily"))
    r = searcher.search("needs a key")
    assert not r.ok and "missing API key" in r.error


def test_duckduckgo_default_when_no_key(settings):
    assert WebSearcher(settings).provider_order() == ["duckduckgo"]


def test_ddgs_backend_normalises_provider_fields(monkeypatch):
    class FakeDDGS:
        def __init__(self, timeout=5):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def text(self, query, max_results=5):
            return [{"title": "T", "href": "https://x.org", "body": "B"}]

    fake_module = types.ModuleType("ddgs")
    fake_module.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_module)
    assert duckduckgo_backend("q", 3, 5.0) == [{"title": "T", "url": "https://x.org", "snippet": "B"}]


def test_api_key_never_appears_in_search_errors(settings):
    secret = FAKE_TAVILY_KEY

    def tavily(q, n, t):
        raise RuntimeError(f"401 Unauthorized for key {secret}")

    cfg = replace(settings, tavily_api_key=secret, search_provider="tavily")
    r = WebSearcher(cfg, backends={"tavily": tavily}).search("auth failure test")
    assert not r.ok and secret not in r.error
