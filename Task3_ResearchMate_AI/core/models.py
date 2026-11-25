"""Structured tool output shared by all tools and the agent."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.validators import wrap_untrusted


class Source(BaseModel):
    title: str
    url: str | None = None
    kind: Literal["web", "file"] = "web"


class ToolResult(BaseModel):
    """What every tool returns (never a bare string or an exception)."""

    tool: str
    ok: bool = True
    content: str = ""
    summary: str = ""           # one-line description shown in the console trace
    error: str | None = None
    sources: list[Source] = Field(default_factory=list)
    untrusted: bool = False     # True when content comes from the web or a file
    truncated: bool = False
    flagged: bool = False       # untrusted content looked like a prompt injection

    @classmethod
    def success(cls, tool: str, content: str, summary: str, **kwargs) -> "ToolResult":
        return cls(tool=tool, ok=True, content=content, summary=summary, **kwargs)

    @classmethod
    def failure(cls, tool: str, error: str) -> "ToolResult":
        return cls(tool=tool, ok=False, error=error, summary=f"Failed: {error}")

    def to_llm_text(self) -> str:
        """Text sent back to the LLM as the tool observation."""
        if not self.ok:
            return f"ERROR from {self.tool}: {self.error}"
        body = self.content
        if self.truncated:
            body += "\n[Output truncated to fit size limit]"
        if self.untrusted:
            body = wrap_untrusted(self.tool, body, flagged=self.flagged)
        return body
