"""Application settings loaded from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from core.exceptions import ConfigError
from core.logger import register_secret

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _mask(secret: str) -> str:
    return "set" if secret else "not set"


@dataclass(frozen=True)
class Settings:
    # LLM
    openai_api_key: str = field(default="", repr=False)
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    llm_timeout: int = 60
    # Search
    tavily_api_key: str = field(default="", repr=False)
    search_provider: str = "auto"          # auto | duckduckgo | tavily
    max_search_results: int = 5
    search_timeout: int = 15
    # Files
    workspace_dir: Path = PROJECT_ROOT / "workspace"
    max_file_bytes: int = 100_000
    # Agent limits
    max_iterations: int = 8
    max_tool_calls: int = 12
    tool_timeout: int = 30
    max_output_chars: int = 8_000
    history_turns: int = 10

    @classmethod
    def load(cls, env_file: Path | None = None) -> "Settings":
        load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

        provider = os.getenv("SEARCH_PROVIDER", "auto").strip().lower() or "auto"
        if provider not in {"auto", "duckduckgo", "tavily"}:
            raise ConfigError("SEARCH_PROVIDER must be one of: auto, duckduckgo, tavily.")

        workspace = Path(os.getenv("WORKSPACE_DIR") or "workspace")
        if not workspace.is_absolute():
            workspace = PROJECT_ROOT / workspace

        settings = cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
            openai_base_url=(os.getenv("OPENAI_BASE_URL") or "").strip() or None,
            llm_timeout=_int_env("LLM_TIMEOUT", 60, 5, 600),
            tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
            search_provider=provider,
            max_search_results=_int_env("MAX_SEARCH_RESULTS", 5, 1, 10),
            search_timeout=_int_env("SEARCH_TIMEOUT", 15, 3, 120),
            workspace_dir=workspace.resolve(),
            max_file_bytes=_int_env("MAX_FILE_BYTES", 100_000, 1_000, 5_000_000),
            max_iterations=_int_env("MAX_ITERATIONS", 8, 1, 25),
            max_tool_calls=_int_env("MAX_TOOL_CALLS", 12, 1, 50),
            tool_timeout=_int_env("TOOL_TIMEOUT", 30, 1, 300),
            max_output_chars=_int_env("MAX_OUTPUT_CHARS", 8_000, 500, 50_000),
            history_turns=_int_env("HISTORY_TURNS", 10, 0, 50),
        )
        register_secret(settings.openai_api_key)
        register_secret(settings.tavily_api_key)
        return settings

    @property
    def llm_ready(self) -> bool:
        return bool(self.openai_api_key)

    def describe(self) -> dict[str, str]:
        """Safe-to-print summary (secrets are only reported as set / not set)."""
        return {
            "LLM model": self.openai_model,
            "OPENAI_API_KEY": _mask(self.openai_api_key),
            "OPENAI_BASE_URL": self.openai_base_url or "default (api.openai.com)",
            "TAVILY_API_KEY": _mask(self.tavily_api_key),
            "Search provider": self.search_provider,
            "Workspace": str(self.workspace_dir),
            "Max iterations": str(self.max_iterations),
            "Tool timeout (s)": str(self.tool_timeout),
        }
