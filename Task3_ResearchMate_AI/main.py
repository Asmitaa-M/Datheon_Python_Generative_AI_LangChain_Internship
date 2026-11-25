"""ResearchMate AI - interactive console entry point.

Usage:
    python main.py                      interactive session
    python main.py -q "your question"   answer one question and exit
    python main.py --check              offline self-check (no API key needed)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from agent.agent import ResearchAgent, build_llm
from agent.executor import ToolExecutor
from config.settings import Settings
from core.console import Console
from core.exceptions import ResearchMateError
from core.logger import redact
from tools import build_tools
from tools.calculator import calculate
from tools.file_reader import FileReader

EXIT_COMMANDS = {"/exit", "/quit"}


def handle_command(command: str, agent: ResearchAgent, console: Console, tools: list) -> bool:
    """Run a slash command. Returns True if the application should exit."""
    name = command.strip().lower()
    if name in EXIT_COMMANDS:
        console.info("Goodbye!")
        return True
    if name == "/help":
        console.help()
    elif name == "/tools":
        console.tools(tools)
    elif name == "/clear":
        agent.reset()
        console.clear_screen()
        console.info("Conversation history cleared.")
    else:
        console.error(f"Unknown command '{command}'. Type /help for the list of commands.")
    return False


def run_query(agent: ResearchAgent, console: Console, query: str) -> bool:
    """Run one query and print the result. Returns True on success."""
    console.query(query)
    response = agent.run(query)
    console.final(response)
    return response.error is None


def self_check(settings: Settings, console: Console) -> int:
    """Offline diagnostics: verifies configuration and the two local tools."""
    console.banner()
    console.label("CONFIGURATION", "cyan")
    for key, value in settings.describe().items():
        console.out(f"  {key}: {value}")

    ok = True
    console.label("OFFLINE TOOL CHECKS", "cyan")
    calc = calculate("sqrt(144) + 2**3")
    console.out(f"  calculator : {'OK' if calc.ok and '20' in calc.content else 'FAIL'}  ({calc.summary})")
    ok &= calc.ok

    reader = FileReader(settings.workspace_dir, settings.max_file_bytes)
    files = reader.available_files()
    console.out(f"  read_file  : {'OK' if files else 'WARN'}  (workspace files: {', '.join(files) or 'none'})")
    escape = reader.read("../../etc/passwd")
    console.out(f"  path guard : {'OK' if not escape.ok else 'FAIL'}  (traversal attempt blocked)")
    ok &= not escape.ok

    names = [t.name for t in build_tools(settings)]
    console.out(f"  registered : {', '.join(names)}")
    console.out(f"  LLM ready  : {'yes' if settings.llm_ready else 'NO - set OPENAI_API_KEY in .env to chat'}")
    console.out("  web search : uses " + ("Tavily, then DuckDuckGo" if settings.tavily_api_key else "DuckDuckGo (no key needed)")
                + " - requires internet access")
    return 0 if ok else 1


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="researchmate", description="ResearchMate AI - autonomous research agent")
    parser.add_argument("-q", "--query", help="answer a single question and exit")
    parser.add_argument("--workspace", type=Path, help="folder the file reader may access (default: ./workspace)")
    parser.add_argument("--check", action="store_true", help="run an offline self-check and exit")
    parser.add_argument("--no-color", action="store_true", help="disable coloured output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console(color=False if args.no_color else None)

    try:
        settings = Settings.load()
        if args.workspace:
            settings = replace(settings, workspace_dir=args.workspace.resolve())
    except ResearchMateError as exc:
        console.error(str(exc))
        return 2

    if args.check:
        return self_check(settings, console)

    console.banner()
    if not settings.llm_ready:
        console.error("OPENAI_API_KEY is not set.")
        console.info("Copy .env.example to .env, add your key, and run again. "
                     "Use 'python main.py --check' to verify the offline parts without a key.")
        return 1

    try:
        llm = build_llm(settings)
        tools = build_tools(settings)
        agent = ResearchAgent(llm, ToolExecutor(tools, settings), settings, on_event=console.trace)
    except Exception as exc:  # configuration / import problems
        console.error(redact(exc))
        return 2

    if args.query:
        return 0 if run_query(agent, console, args.query) else 1

    console.instructions([t.name for t in tools])
    while True:
        try:
            line = input(console.paint("\nYou> ", "bold")).strip()
        except (EOFError, KeyboardInterrupt):
            console.out()
            console.info("Goodbye!")
            return 0
        if not line:
            continue
        if line.startswith("/"):
            if handle_command(line, agent, console, tools):
                return 0
            continue
        try:
            run_query(agent, console, line)
        except KeyboardInterrupt:
            console.error("Interrupted. Type /exit to quit.")
        except Exception as exc:  # last-resort guard: never crash the session
            console.error(f"Unexpected error: {redact(exc)}")


if __name__ == "__main__":
    sys.exit(main())
