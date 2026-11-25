"""Shared fixtures: a scripted fake LLM and a temporary workspace. No network, no API keys."""
from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent.agent import ResearchAgent, TraceEvent
from agent.executor import ToolExecutor
from config.settings import Settings
from tools import build_tools
from tools.web_search import WebSearcher

_ids = itertools.count(1)

# Built at runtime so no key-shaped literal appears in the source tree.
FAKE_OPENAI_KEY = "sk-" + "FAKEKEY" * 4
FAKE_TAVILY_KEY = "tvly-" + "FAKEKEY" * 3


def tool_call(name: str, **args: Any) -> dict:
    return {"name": name, "args": args, "id": f"call_{next(_ids)}", "type": "tool_call"}


def ai_calls(*calls: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls))


def ai_text(text: str) -> AIMessage:
    return AIMessage(content=text)


class ScriptedChatModel(BaseChatModel):
    """Returns pre-scripted AI messages in order; records every prompt it receives.

    An item that is an Exception instance is raised instead, to simulate LLM outages.
    """

    script: list[Any]
    seen: list[list[BaseMessage]] = []
    bound_tools: list[Any] = []
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: D401 - mimic the real API
        self.bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        if self.cursor >= len(self.script):
            raise AssertionError("ScriptedChatModel ran out of scripted responses")
        item = self.script[self.cursor]
        self.cursor += 1
        if isinstance(item, Exception):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "expenses.csv").write_text(
        "date,category,description,amount\n"
        "2026-10-01,Housing,Rent,1200.00\n"
        "2026-10-03,Groceries,Shop,86.40\n"
        "2026-10-09,Dining,Dinner,58.90\n",
        encoding="utf-8",
    )
    (ws / "notes.txt").write_text("Topics: agentic AI; RAG; small language models\n", encoding="utf-8")
    (ws / "data.json").write_text('{"a": 1, "b": [1, 2, 3]}', encoding="utf-8")
    (ws / "readme.md").write_text("# Title\n\nSome *markdown*.\n", encoding="utf-8")
    (ws / "script.py").write_text("print('hello')\n", encoding="utf-8")
    return ws


@pytest.fixture
def settings(workspace):
    return replace(
        Settings(),
        openai_api_key="",
        tavily_api_key="",
        workspace_dir=workspace.resolve(),
        max_iterations=5,
        tool_timeout=2,
        max_file_bytes=5_000,
    )


FAKE_SEARCH_RESULTS = [
    {"title": "India population - Example Stats", "url": "https://example.com/india", "snippet": "India has about 1.45 billion people."},
    {"title": "Japan population - Example Stats", "url": "https://example.com/japan", "snippet": "Japan has about 123 million people."},
]


@pytest.fixture
def fake_search():
    def backend(query, max_results, timeout):
        return list(FAKE_SEARCH_RESULTS)

    return backend


@pytest.fixture
def make_agent(settings, fake_search):
    """Factory: make_agent(script, settings_override=None) -> (agent, llm, events)."""

    def _make(script, search_backend=None, **overrides):
        cfg = replace(settings, **overrides) if overrides else settings
        searcher = WebSearcher(cfg, backends={"duckduckgo": search_backend or fake_search})
        executor = ToolExecutor(build_tools(cfg, searcher), cfg)
        events: list[TraceEvent] = []
        llm = ScriptedChatModel(script=script, seen=[], bound_tools=[])
        agent = ResearchAgent(llm, executor, cfg, on_event=events.append)
        return agent, llm, events

    return _make
