"""Live integration tests. They need real credentials / internet, so they are skipped by default.

Run with:   RUN_LIVE_TESTS=1 pytest tests/test_live.py      (PowerShell: $env:RUN_LIVE_TESTS=1)
"""
import os

import pytest

from agent.agent import ResearchAgent, build_llm
from agent.executor import ToolExecutor
from config.settings import Settings
from tools import build_tools
from tools.web_search import WebSearcher

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to run live tests"),
]


def test_live_duckduckgo_search():
    settings = Settings.load()
    result = WebSearcher(settings).search("Python programming language")
    assert result.ok and result.sources


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_live_agent_no_tool_needed():
    settings = Settings.load()
    agent = ResearchAgent(build_llm(settings), ToolExecutor(build_tools(settings), settings), settings)
    response = agent.run("Explain what an AI agent is in two sentences.")
    assert response.answer and response.tool_calls == 0


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_live_agent_file_plus_calculator():
    settings = Settings.load()
    agent = ResearchAgent(build_llm(settings), ToolExecutor(build_tools(settings), settings), settings)
    response = agent.run("Read expenses.csv, calculate the total of all expenses and identify the highest one.")
    assert response.tool_calls >= 2 and "1,200" in response.answer.replace(" ", "")
