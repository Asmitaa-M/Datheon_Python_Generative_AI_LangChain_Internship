# Test Execution Report

**Environment:** Linux, Python 3.13, fresh virtualenv built from `requirements.txt`
**Command:** `pytest`
**Result:** `204 passed, 3 skipped` (0 failed) in ~2 s - also run with `-W error::DeprecationWarning`: same result

| Test file | Tests | Covers |
|---|---:|---|
| `test_calculator.py` | 79 | all operators, functions, aggregates, div-by-zero, invalid syntax, code-injection attempts, size limits |
| `test_file_reader.py` | 28 | txt/csv/json/md/py, missing file, unsupported types, permission/OS errors, traversal, symlink escape, binary, truncation |
| `test_web_search.py` | 26 | success, result limit, dedupe/unsafe URLs, empty, network error, timeout, rate limit, invalid queries, provider fallback, key redaction |
| `test_agent.py` | 24 | direct answer, single/multi/parallel tools, compound (file+calc, all three), invalid args, unknown tool, tool crash, tool timeout, max iterations, duplicate-call and budget guards, LLM failure, history limit |
| `test_security.py` | 39 | calculator injection, path traversal, malicious file content, search-result injection, delimiter forging, output limits, API-key leakage, repo secret scan |
| `test_cli.py` | 6 | slash commands, trace format, redaction in trace, no-key start-up, `--check`, exit codes |
| `test_openai_wire.py` | 2 | **real** `ChatOpenAI` client against a local OpenAI-compatible server: tool schemas, `tool_calls` parsing, tool-result round trip, key only in auth header |
| `test_live.py` | 3 (skipped) | live DuckDuckGo search, live agent without tools, live agent file+calculator |

## Additional validation performed

- All 14 modules import with warnings treated as errors; no use of `AgentExecutor`, `create_tool_calling_agent`, `langchain-community` or other legacy APIs.
- `main.py --check` runs offline; `main.py` without a key exits cleanly with instructions.
- End-to-end run of `main.py -q` through the real OpenAI client and real tools against a local stub model server; total/max independently recomputed (2,055.82 / 1,200.00).
- No key-shaped strings in the repository; `.env` git-ignored; `.env.example` contains no values.

## Not verified (requires credentials / internet)

| Item | Why | How to verify |
|---|---|---|
| Live LLM tool-selection quality | No `OPENAI_API_KEY` available | `RUN_LIVE_TESTS=1 pytest tests/test_live.py`, then try the sample queries |
| Live DuckDuckGo / Tavily search | Sandbox network allowlist blocked search hosts (live test fails there with a handled "network error") | same live test, on a machine with internet |
| Windows / macOS, Python 3.10-3.12 | Only Linux + Python 3.13 available | run `pytest` on those platforms |
