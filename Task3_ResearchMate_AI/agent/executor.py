"""Guarded tool execution: validation, timeout, error handling, output limits."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from langchain_core.tools import BaseTool
from pydantic import ValidationError

from config.settings import Settings
from core.logger import get_logger, redact
from core.models import ToolResult
from core.validators import detect_injection, truncate

log = get_logger(__name__)


class ToolExecutor:
    """Runs one tool call safely. ``run`` never raises."""

    def __init__(self, tools: list[BaseTool], settings: Settings):
        self.tools = {t.name: t for t in tools}
        self.settings = settings

    @property
    def tool_names(self) -> list[str]:
        return list(self.tools)

    def run(self, name: str, args: object) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult.failure(
                name or "unknown", f"Unknown tool '{name}'. Available tools: {', '.join(self.tools)}."
            )
        if not isinstance(args, dict):
            return ToolResult.failure(name, "Tool arguments must be a JSON object.")

        # 1. Validate arguments against the tool's pydantic schema.
        try:
            validated = tool.args_schema.model_validate(args).model_dump()
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}" for e in exc.errors()
            )
            return ToolResult.failure(name, f"Invalid arguments - {problems}")

        # 2. Execute with a timeout so a hung tool cannot block the agent.
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(tool.invoke, validated)
        try:
            result = future.result(timeout=self.settings.tool_timeout)
        except FutureTimeout:
            log.warning("Tool %s timed out after %ss", name, self.settings.tool_timeout)
            return ToolResult.failure(name, f"Tool timed out after {self.settings.tool_timeout} seconds.")
        except Exception as exc:  # defensive: tools are expected to return ToolResult
            log.exception("Tool %s crashed", name)
            return ToolResult.failure(name, f"Tool failed unexpectedly ({type(exc).__name__}).")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        if not isinstance(result, ToolResult):
            return ToolResult.failure(name, "Tool returned an invalid result.")
        return self._finalize(result)

    def _finalize(self, result: ToolResult) -> ToolResult:
        """Apply output size limit and prompt-injection screening."""
        result.content, cut = truncate(result.content, self.settings.max_output_chars)
        result.truncated = result.truncated or cut
        if result.error:
            result.error = redact(result.error)
        if result.untrusted and detect_injection(result.content):
            result.flagged = True
            log.warning("Possible prompt injection detected in %s output", result.tool)
        return result
