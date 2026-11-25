"""Exercises the REAL langchain-openai client against a local OpenAI-compatible HTTP server.

No internet and no API key: the server runs on 127.0.0.1 and plays the model's role.
This verifies the real wire format - tool schemas sent to the API, tool_calls parsing,
tool-result round trip - which the in-process fake LLM cannot.
"""
from __future__ import annotations

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent.agent import ResearchAgent, build_llm
from agent.executor import ToolExecutor
from tools import build_tools


def _completion(message: dict, finish: str) -> dict:
    return {
        "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture
def mock_openai(monkeypatch):
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
            msgs = body["messages"]
            if msgs[-1]["role"] == "tool":   # we have an observation -> produce the final answer
                reply = _completion({"role": "assistant", "content": f"FINAL: {msgs[-1]['content']}"}, "stop")
            else:                            # first turn -> ask for the calculator
                reply = _completion({
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": "call_abc", "type": "function",
                                    "function": {"name": "calculator",
                                                 "arguments": json.dumps({"expression": "1450000000 - 123000000"})}}],
                }, "tool_calls")
            payload = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    yield f"http://127.0.0.1:{server.server_port}/v1", requests
    server.shutdown()


def test_real_openai_client_full_tool_loop(settings, mock_openai):
    base_url, requests = mock_openai
    cfg = replace(settings, openai_api_key="test-key-not-real", openai_base_url=base_url, openai_model="test-model", llm_timeout=10)
    events = []
    agent = ResearchAgent(build_llm(cfg), ToolExecutor(build_tools(cfg), cfg), cfg, on_event=events.append)

    response = agent.run("What is 1450000000 minus 123000000?")

    assert response.tool_calls == 1 and not response.error
    assert "1,327,000,000" in response.answer
    assert [e.tool for e in events if e.kind == "action"] == ["calculator"]

    first, second = requests[0]["body"], requests[1]["body"]
    # tool schemas were sent in OpenAI function-calling format
    sent = {t["function"]["name"]: t["function"] for t in first["tools"]}
    assert set(sent) == {"web_search", "calculator", "read_file"}
    assert sent["calculator"]["parameters"]["required"] == ["expression"]
    assert sent["web_search"]["parameters"]["required"] == ["query"]
    # the API key travels only in the Authorization header, never in the body or the answer
    assert requests[0]["auth"] == "Bearer test-key-not-real"
    assert "test-key-not-real" not in json.dumps(first) and "test-key-not-real" not in response.answer
    # the tool observation was returned to the model with the matching call id
    tool_msg = second["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_abc"
    assistant_msg = second["messages"][-2]
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "calculator"
    # system prompt tells the model to treat external content as untrusted data
    assert first["messages"][0]["role"] == "system" and "UNTRUSTED" in first["messages"][0]["content"]


def test_build_llm_requires_key(settings):
    from core.exceptions import LLMError

    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        build_llm(replace(settings, openai_api_key=""))
