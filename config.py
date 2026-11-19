"""Application settings loaded from environment variables / a ``.env`` file.

Nothing secret is ever hardcoded here. The API key only comes from the
environment, and it is hidden from ``repr()`` so it cannot leak into logs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from dotenv import load_dotenv

from exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "gemini-2.5-flash"

# Values people commonly leave in .env.example; they are not real keys.
_PLACEHOLDER_API_KEYS = frozenset(
    {"", "your_api_key_here", "your-api-key-here", "changeme", "xxx"}
)
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the extractor."""

    api_key: Optional[str] = field(default=None, repr=False)
    model: str = DEFAULT_MODEL
    mock_mode: bool = False
    max_retries: int = 3  # correction attempts after the first try
    api_max_retries: int = 2  # retries for rate limits / network errors
    api_retry_delay: float = 2.0  # seconds; doubles after each API retry
    temperature: float = 0.0
    timeout_seconds: float = 60.0
    output_dir: Path = PROJECT_ROOT / "output"

    @property
    def has_api_key(self) -> bool:
        """Return True when a non-placeholder API key is configured."""
        return bool(self.api_key) and self.api_key.strip().lower() not in (
            _PLACEHOLDER_API_KEYS
        )


def _parse_bool(raw: str, name: str) -> bool:
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ConfigurationError(
        f"{name} must be true or false (got {raw!r})."
    )


def _parse_int(raw: str, name: str, minimum: int, maximum: int) -> int:
    try:
        value = int(raw.strip())
    except ValueError:
        raise ConfigurationError(
            f"{name} must be a whole number (got {raw!r})."
        ) from None
    if not minimum <= value <= maximum:
        raise ConfigurationError(
            f"{name} must be between {minimum} and {maximum} (got {value})."
        )
    return value


def _parse_float(raw: str, name: str, minimum: float, maximum: float) -> float:
    try:
        value = float(raw.strip())
    except ValueError:
        raise ConfigurationError(
            f"{name} must be a number (got {raw!r})."
        ) from None
    if not minimum <= value <= maximum:
        raise ConfigurationError(
            f"{name} must be between {minimum} and {maximum} (got {value})."
        )
    return value


def load_settings(
    env: Optional[Mapping[str, str]] = None,
    dotenv_path: Optional[Path] = None,
) -> Settings:
    """Build :class:`Settings` from the environment.

    Args:
        env: Mapping to read from. When ``None`` (normal use), the project's
            ``.env`` file is loaded first and ``os.environ`` is used. Passing a
            mapping skips ``.env`` loading, which keeps tests isolated.
        dotenv_path: Optional alternative location of the ``.env`` file.

    Raises:
        ConfigurationError: If a setting has an invalid value.
    """
    if env is None:
        # override=False: real environment variables win over the .env file.
        load_dotenv(dotenv_path or PROJECT_ROOT / ".env", override=False)
        env = os.environ

    defaults = Settings()
    api_key = env.get("GEMINI_API_KEY", "").strip() or None

    output_dir_raw = env.get("OUTPUT_DIR", "").strip()
    return Settings(
        api_key=api_key,
        model=env.get("GEMINI_MODEL", "").strip() or defaults.model,
        mock_mode=_parse_bool(env.get("LLM_MOCK_MODE", "false"), "LLM_MOCK_MODE"),
        max_retries=_parse_int(
            env.get("MAX_RETRIES", str(defaults.max_retries)), "MAX_RETRIES", 0, 10
        ),
        api_max_retries=_parse_int(
            env.get("API_MAX_RETRIES", str(defaults.api_max_retries)),
            "API_MAX_RETRIES",
            0,
            10,
        ),
        api_retry_delay=_parse_float(
            env.get("API_RETRY_DELAY", str(defaults.api_retry_delay)),
            "API_RETRY_DELAY",
            0.0,
            300.0,
        ),
        temperature=_parse_float(
            env.get("LLM_TEMPERATURE", str(defaults.temperature)),
            "LLM_TEMPERATURE",
            0.0,
            2.0,
        ),
        timeout_seconds=_parse_float(
            env.get("LLM_TIMEOUT_SECONDS", str(defaults.timeout_seconds)),
            "LLM_TIMEOUT_SECONDS",
            1.0,
            600.0,
        ),
        output_dir=Path(output_dir_raw) if output_dir_raw else defaults.output_dir,
    )
