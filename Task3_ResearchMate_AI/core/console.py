"""Console rendering: banner, execution trace, answers. Pure presentation, no logic."""
from __future__ import annotations

import os
import sys

from agent.agent import AgentResponse, TraceEvent
from core.logger import redact

WIDTH = 50
RULE = "=" * WIDTH

_COLORS = {
    "bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33",
    "blue": "34", "magenta": "35", "cyan": "36",
}


def _enable_windows_ansi() -> bool:
    """Turn on ANSI escape processing in Windows 10+ consoles."""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def _short(text: object, limit: int = 300) -> str:
    text = redact(" ".join(str(text).split()))
    return text if len(text) <= limit else text[: limit - 3] + "..."


class Console:
    def __init__(self, color: bool | None = None):
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
            except Exception:
                pass
        if color is None:
            color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
        if color and os.name == "nt":
            color = _enable_windows_ansi()
        self.color = color

    # ------------------------------------------------------------ helpers
    def paint(self, text: str, *styles: str) -> str:
        if not self.color or not styles:
            return text
        codes = ";".join(_COLORS[s] for s in styles)
        return f"\033[{codes}m{text}\033[0m"

    def out(self, text: str = "") -> None:
        print(text, flush=True)

    def label(self, name: str, *styles: str) -> None:
        self.out()
        self.out(self.paint(f"{name}:", "bold", *styles))

    def clear_screen(self) -> None:
        if self.color:
            print("\033[2J\033[H", end="", flush=True)

    # ------------------------------------------------------------- screens
    def banner(self) -> None:
        self.out(self.paint(RULE, "cyan"))
        self.out(self.paint("RESEARCHMATE AI".center(WIDTH), "bold", "cyan"))
        self.out(self.paint("AUTONOMOUS RESEARCH AGENT".center(WIDTH), "cyan"))
        self.out(self.paint(RULE, "cyan"))

    def instructions(self, tool_names: list[str]) -> None:
        self.out()
        self.out("Ask a research question. The agent decides by itself which tools to use:")
        self.out("  " + ", ".join(tool_names))
        self.out(self.paint("Commands: /help  /tools  /clear  /exit", "dim"))

    def help(self) -> None:
        self.out()
        self.out("/help   - Show available commands")
        self.out("/tools  - Display registered tools")
        self.out("/clear  - Clear conversation history")
        self.out("/exit   - Exit the application")
        self.out()
        self.out("Example queries:")
        self.out("  Explain what an AI agent is.")
        self.out("  Read expenses.csv, calculate the total and find the highest expense.")
        self.out("  Find the current population of India and Japan and calculate the difference.")

    def tools(self, tools: list) -> None:
        self.out()
        for tool in tools:
            self.out(self.paint(tool.name, "bold", "green"))
            self.out("  " + _short(tool.description, 400))

    # --------------------------------------------------------------- trace
    def query(self, text: str) -> None:
        self.label("USER QUERY", "cyan")
        self.out(_short(text, 1000))

    def trace(self, event: TraceEvent) -> None:
        if event.kind == "analyzing":
            self.label("AGENT", "blue")
            self.out(event.text)
        elif event.kind == "decision":
            self.label("DECISION", "blue")
            self.out(event.text)
        elif event.kind == "action":
            self.label("ACTION", "yellow")
            self.out(f"Calling {event.tool}")
            if event.args:
                self.label("INPUT", "yellow")
                self.out(self._format_args(event.args))
        elif event.kind == "observation":
            self.label("OBSERVATION", "green" if event.ok else "red")
            self.out(_short(event.text))
        elif event.kind == "warning":
            self.label("NOTICE", "magenta")
            self.out(_short(event.text))

    @staticmethod
    def _format_args(args: dict) -> str:
        if len(args) == 1:
            return _short(next(iter(args.values())))
        return _short(", ".join(f"{k}={v}" for k, v in args.items() if v is not None))

    def final(self, response: AgentResponse) -> None:
        style = "red" if response.error else "green"
        self.label("FINAL ANSWER", style)
        self.out(response.answer)
        if response.limit_reached:
            self.out(self.paint("(Stopped at the iteration limit; answer may be incomplete.)", "yellow"))
        if response.sources:
            self.label("SOURCES", "cyan")
            for i, src in enumerate(response.sources, 1):
                if src.kind == "file":
                    self.out(f"{i}. Local file - {src.title}")
                else:
                    self.out(f"{i}. {src.title} - {src.url}")
        self.out()
        self.out(self.paint(RULE, "cyan"))

    def error(self, message: str) -> None:
        self.out(self.paint(f"Error: {redact(message)}", "red"))

    def info(self, message: str) -> None:
        self.out(self.paint(message, "dim"))
