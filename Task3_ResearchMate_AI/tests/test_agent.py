"""Agent tests use a scripted fake LLM, so they verify the loop, the real tools and the
guards - not the model's own judgement (see tests/test_live.py for that)."""
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from tests.conftest import ai_calls, ai_text, tool_call


def kinds(events):
    return [e.kind for e in events]


def test_direct_answer_without_tools(make_agent):
    agent, llm, events = make_agent([ai_text("An AI agent is software that acts toward goals.")])
    r = agent.run("Explain what an AI agent is.")
    assert r.answer.startswith("An AI agent")
    assert r.tool_calls == 0 and r.sources == [] and not r.error
    assert kinds(events) == ["analyzing", "decision"]
    assert len(llm.seen) == 1


def test_tools_are_bound_to_the_llm(make_agent):
    agent, llm, _ = make_agent([ai_text("hi")])
    assert {t.name for t in llm.bound_tools} == {"web_search", "calculator", "read_file"}


def test_single_tool_execution(make_agent):
    agent, llm, events = make_agent([
        ai_calls(tool_call("calculator", expression="1450000000 - 123000000")),
        ai_text("The difference is 1,327,000,000."),
    ])
    r = agent.run("What is 1450000000 minus 123000000?")
    assert r.tool_calls == 1 and "1,327,000,000" in r.answer
    obs = [e for e in events if e.kind == "observation"][0]
    assert obs.ok and "1,327,000,000" in obs.text
    # the observation was really fed back to the LLM
    tool_msgs = [m for m in llm.seen[1] if isinstance(m, ToolMessage)]
    assert "1,327,000,000" in tool_msgs[0].content


def test_multiple_tools_search_then_calculator(make_agent):
    agent, llm, events = make_agent([
        ai_calls(tool_call("web_search", query="current population of India and Japan")),
        ai_calls(tool_call("calculator", expression="1450000000 - 123000000")),
        ai_text("India is larger by about 1,327,000,000 people."),
    ])
    r = agent.run("Compare the population of India and Japan and calculate the difference.")
    actions = [e.tool for e in events if e.kind == "action"]
    assert actions == ["web_search", "calculator"]
    assert r.tool_calls == 2 and r.iterations == 3
    assert [s.url for s in r.sources] == ["https://example.com/india", "https://example.com/japan"]


def test_parallel_tool_calls_in_one_step(make_agent):
    agent, llm, events = make_agent([
        ai_calls(tool_call("read_file", path="notes.txt"), tool_call("calculator", expression="2+2")),
        ai_text("done"),
    ])
    r = agent.run("do both")
    assert r.tool_calls == 2
    assert len([m for m in llm.seen[1] if isinstance(m, ToolMessage)]) == 2


def test_compound_file_reader_plus_calculator(make_agent):
    agent, llm, events = make_agent([
        ai_calls(tool_call("read_file", path="expenses.csv")),
        ai_calls(tool_call("calculator", expression="sum([1200.00, 86.40, 58.90])")),
        ai_calls(tool_call("calculator", expression="max([1200.00, 86.40, 58.90])")),
        ai_text("Total 1,345.3; highest is Rent at 1,200."),
    ])
    r = agent.run("Read expenses.csv, calculate the total and find the highest expense.")
    assert r.tool_calls == 3
    # tools really executed against the real file: the CSV content reached the LLM
    csv_obs = [m for m in llm.seen[1] if isinstance(m, ToolMessage)][0].content
    assert "Housing" in csv_obs and "1200.00" in csv_obs
    calc_obs = [m for m in llm.seen[2] if isinstance(m, ToolMessage)][-1].content
    assert "1,345.3" in calc_obs
    assert [s.kind for s in r.sources] == ["file"]


def test_compound_all_three_tools(make_agent):
    agent, llm, events = make_agent([
        ai_calls(tool_call("read_file", path="expenses.csv")),
        ai_calls(tool_call("calculator", expression="mean([1200, 86.4, 58.9])")),
        ai_calls(tool_call("web_search", query="average household monthly spending benchmark")),
        ai_text("Your average is 448.43; benchmarks differ."),
    ])
    r = agent.run("Read expenses.csv, average it, and search for benchmarks.")
    assert [e.tool for e in events if e.kind == "action"] == ["read_file", "calculator", "web_search"]
    assert {s.kind for s in r.sources} == {"file", "web"}
    assert all(e.ok for e in events if e.kind == "observation")


def test_conversation_history_is_sent_and_limited(make_agent):
    agent, llm, _ = make_agent([ai_text(f"answer {i}") for i in range(4)], history_turns=2)
    for i in range(4):
        agent.run(f"question {i}")
    assert len(agent.history) == 4  # 2 turns x (human + ai)
    last_prompt = llm.seen[-1]
    assert isinstance(last_prompt[0], SystemMessage)
    assert [m.content for m in last_prompt if isinstance(m, HumanMessage)] == ["question 1", "question 2", "question 3"]


def test_reset_clears_history(make_agent):
    agent, _, _ = make_agent([ai_text("a")])
    agent.run("q")
    assert agent.history
    agent.reset()
    assert agent.history == []


def test_invalid_tool_arguments_are_reported_to_llm_and_recovered(make_agent):
    agent, llm, events = make_agent([
        ai_calls(tool_call("calculator")),                                   # missing argument
        ai_calls(tool_call("calculator", expression=12345)),                  # wrong type
        ai_calls(tool_call("web_search", query="x")),                         # too short
        ai_calls(tool_call("calculator", expression="2+2")),                  # corrected
        ai_text("It is 4."),
    ])
    r = agent.run("what is 2+2")
    obs = [e for e in events if e.kind == "observation"]
    assert [o.ok for o in obs] == [False, False, False, True]
    assert "Invalid arguments" in obs[0].text
    assert r.answer == "It is 4." and not r.error


def test_unknown_tool_name_is_handled(make_agent):
    agent, llm, events = make_agent([ai_calls(tool_call("delete_files", path="/")), ai_text("sorry")])
    r = agent.run("do something bad")
    obs = [e for e in events if e.kind == "observation"][0]
    assert not obs.ok and "Unknown tool" in obs.text
    assert r.answer == "sorry"


def test_tool_failure_is_fed_back_not_raised(make_agent):
    def broken(q, n, t):
        raise ConnectionError("offline")

    agent, llm, events = make_agent(
        [ai_calls(tool_call("web_search", query="anything here")), ai_text("Search is unavailable right now.")],
        search_backend=broken,
    )
    r = agent.run("search something")
    obs = [e for e in events if e.kind == "observation"][0]
    assert not obs.ok
    assert "Search is unavailable" in r.answer and not r.error


def test_unexpected_tool_exception_is_contained(make_agent):
    def explode(q, n, t):
        raise ValueError("boom")

    # ValueError inside the backend is wrapped by the searcher; crash the tool itself instead
    agent, llm, events = make_agent([ai_calls(tool_call("read_file", path="notes.txt")), ai_text("ok")])
    original = agent.executor.tools["read_file"].func

    def crash(path):
        raise RuntimeError("tool crashed")

    agent.executor.tools["read_file"].func = crash
    try:
        r = agent.run("read it")
    finally:
        agent.executor.tools["read_file"].func = original
    obs = [e for e in events if e.kind == "observation"][0]
    assert not obs.ok and "failed unexpectedly" in obs.text
    assert r.answer == "ok"


def test_tool_timeout(make_agent):
    def slow(q, n, t):
        time.sleep(3)
        return []

    agent, llm, events = make_agent(
        [ai_calls(tool_call("web_search", query="slow query")), ai_text("timed out, sorry")],
        search_backend=slow, tool_timeout=1,
    )
    start = time.monotonic()
    r = agent.run("search")
    assert time.monotonic() - start < 2.5
    obs = [e for e in events if e.kind == "observation"][0]
    assert not obs.ok and "timed out" in obs.text
    assert r.answer == "timed out, sorry"


def test_maximum_iterations_forces_final_answer(make_agent):
    script = [ai_calls(tool_call("calculator", expression=f"{i}+1")) for i in range(3)]
    script.append(ai_text("Best effort answer after hitting the limit."))
    agent, llm, events = make_agent(script, max_iterations=3)
    r = agent.run("loop forever")
    assert r.limit_reached and r.iterations == 3 and r.tool_calls == 3
    assert "Best effort" in r.answer
    assert any(e.kind == "warning" and "maximum" in e.text for e in events)
    # the final call was made with a "no more tools" instruction
    assert "limit reached" in llm.seen[-1][-1].content.lower()


def test_duplicate_calls_are_blocked_to_prevent_loops(make_agent):
    same = {"expression": "2+2"}
    agent, llm, events = make_agent([
        ai_calls(tool_call("calculator", **same)),
        ai_calls(tool_call("calculator", **same)),
        ai_text("4"),
    ])
    r = agent.run("what is 2+2")
    obs = [e for e in events if e.kind == "observation"]
    assert obs[0].ok and not obs[1].ok and "Duplicate" in obs[1].text
    assert r.answer == "4"


def test_tool_call_budget_is_enforced(make_agent):
    script = [ai_calls(tool_call("calculator", expression=f"{i}*2")) for i in range(4)] + [ai_text("stopped")]
    agent, llm, events = make_agent(script, max_tool_calls=2, max_iterations=6)
    r = agent.run("many calls")
    obs = [e for e in events if e.kind == "observation"]
    assert [o.ok for o in obs] == [True, True, False, False]
    assert "budget" in obs[2].text


def test_too_many_parallel_calls_capped(make_agent):
    calls = [tool_call("calculator", expression=f"{i}+1") for i in range(6)]
    agent, llm, events = make_agent([ai_calls(*calls), ai_text("done")], max_tool_calls=20)
    r = agent.run("parallel")
    obs = [e for e in events if e.kind == "observation"]
    assert [o.ok for o in obs] == [True] * 4 + [False] * 2
    # every tool_call id still receives a ToolMessage (required by the API)
    assert len([m for m in llm.seen[1] if isinstance(m, ToolMessage)]) == 6


def test_llm_failure_is_graceful_and_history_unchanged(make_agent):
    agent, llm, events = make_agent([RuntimeError("connection refused")])
    r = agent.run("hello")
    assert r.error and "could not get a response" in r.answer
    assert agent.history == []


def test_llm_failure_mid_run_keeps_collected_sources(make_agent):
    agent, llm, _ = make_agent([
        ai_calls(tool_call("web_search", query="population of Japan")),
        RuntimeError("503 service unavailable"),
    ])
    r = agent.run("search then fail")
    assert r.error and len(r.sources) == 2


def test_empty_llm_answer_gets_fallback_text(make_agent):
    agent, _, _ = make_agent([ai_text("")])
    r = agent.run("hello")
    assert "not able to produce an answer" in r.answer


def test_malformed_tool_call_json_triggers_retry(make_agent):
    from langchain_core.messages import AIMessage

    bad = AIMessage(content="", invalid_tool_calls=[
        {"name": "calculator", "args": "{not json", "id": "bad1", "error": "bad json", "type": "invalid_tool_call"}
    ])
    agent, llm, events = make_agent([bad, ai_calls(tool_call("calculator", expression="1+1")), ai_text("2")])
    r = agent.run("1+1")
    assert r.answer == "2"
    msgs = [m for m in llm.seen[1] if isinstance(m, ToolMessage)]
    assert "not valid JSON" in msgs[0].content


def test_invalid_user_input_rejected_before_llm(make_agent):
    agent, llm, _ = make_agent([ai_text("never used")])
    for bad in ["", "   ", "x" * 5000]:
        r = agent.run(bad)
        assert r.error
    assert llm.seen == []


def test_final_response_includes_deduplicated_sources(make_agent):
    agent, llm, _ = make_agent([
        ai_calls(tool_call("web_search", query="first search")),
        ai_calls(tool_call("web_search", query="second search")),
        ai_text("done"),
    ])
    r = agent.run("search twice")
    assert len(r.sources) == 2  # same two URLs from both searches collapse
