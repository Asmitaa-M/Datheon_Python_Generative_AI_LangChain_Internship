import main as app
from core.console import Console
from agent.agent import AgentResponse, TraceEvent
from core.models import Source
from tests.conftest import ai_text


def test_commands(make_agent, capsys):
    agent, _, _ = make_agent([ai_text("hello")])
    console = Console(color=False)
    tools = list(agent.executor.tools.values())
    agent.run("hi")
    assert agent.history

    assert app.handle_command("/help", agent, console, tools) is False
    assert "/clear" in capsys.readouterr().out
    assert app.handle_command("/tools", agent, console, tools) is False
    out = capsys.readouterr().out
    assert "web_search" in out and "calculator" in out and "read_file" in out
    assert app.handle_command("/clear", agent, console, tools) is False
    assert agent.history == []
    assert app.handle_command("/bogus", agent, console, tools) is False
    assert "Unknown command" in capsys.readouterr().out
    assert app.handle_command("/exit", agent, console, tools) is True
    assert app.handle_command("/QUIT", agent, console, tools) is True


def test_trace_rendering_matches_spec(capsys):
    c = Console(color=False)
    c.query("Compare populations")
    c.trace(TraceEvent(kind="analyzing", text="Analyzing the query..."))
    c.trace(TraceEvent(kind="action", tool="calculator", args={"expression": "1 - 1"}))
    c.trace(TraceEvent(kind="observation", tool="calculator", text="1 - 1 = 0"))
    c.final(AgentResponse(answer="Zero.", sources=[Source(title="Site", url="https://a.com"), Source(title="x.csv", kind="file")]))
    out = capsys.readouterr().out
    for label in ["USER QUERY:", "AGENT:", "ACTION:", "Calling calculator", "INPUT:", "OBSERVATION:", "FINAL ANSWER:", "SOURCES:"]:
        assert label in out
    assert "1. Site - https://a.com" in out and "2. Local file - x.csv" in out


def test_trace_redacts_secrets_in_inputs(capsys):
    Console(color=False).trace(TraceEvent(kind="action", tool="web_search", args={"query": "key sk-abc123def456ghi789"}))
    assert "sk-abc123" not in capsys.readouterr().out


def test_one_shot_query_exit_code_reflects_agent_outcome(monkeypatch, make_agent, capsys):
    ok_agent, _, _ = make_agent([ai_text("fine")])
    bad_agent, _, _ = make_agent([RuntimeError("provider down")])
    console = Console(color=False)
    assert app.run_query(ok_agent, console, "hello") is True
    assert app.run_query(bad_agent, console, "hello") is False
    assert "could not get a response" in capsys.readouterr().out


def test_no_key_exits_cleanly(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(app.Settings, "load", classmethod(lambda cls, env_file=None: cls()))
    assert app.main(["--no-color"]) == 1
    assert "OPENAI_API_KEY is not set" in capsys.readouterr().out


def test_self_check_runs_offline(monkeypatch, capsys):
    monkeypatch.setattr(app.Settings, "load", classmethod(lambda cls, env_file=None: cls()))
    assert app.main(["--check", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "calculator : OK" in out and "path guard : OK" in out
