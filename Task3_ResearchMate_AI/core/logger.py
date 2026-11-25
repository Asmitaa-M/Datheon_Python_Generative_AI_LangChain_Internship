"""Logging with automatic secret redaction.

Every log record passes through RedactingFilter, so API keys can never leak
into log output even if a library puts one in an exception message.
"""
from __future__ import annotations

import logging
import os
import re
import sys

_REDACTED = "[REDACTED]"

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-\*]{3,}"),              # OpenAI-style keys (also masked forms)
    re.compile(r"tvly-[A-Za-z0-9_\-\*]{3,}"),            # Tavily keys
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),   # Authorization headers
    re.compile(r"(?i)\b(api[_-]?key|token|secret|authorization)\b(\s*[=:]\s*)\S+"),
]

_registered_secrets: set[str] = set()


def register_secret(value: str | None) -> None:
    """Remember an exact secret value so it is always scrubbed from output."""
    if value and len(value) >= 6:
        _registered_secrets.add(value)


def redact(text: object) -> str:
    """Return ``text`` with API keys and registered secrets removed."""
    out = str(text)
    for secret in _registered_secrets:
        out = out.replace(secret, _REDACTED)
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            out = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}{_REDACTED}", out)
        else:
            out = pattern.sub(_REDACTED, out)
    return out


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        if record.exc_info:
            # Render the traceback now so it is redacted too.
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        return True


def get_logger(name: str = "researchmate") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler.addFilter(RedactingFilter())
        logger.addHandler(handler)
        level = os.getenv("LOG_LEVEL", "WARNING").upper()
        logger.setLevel(getattr(logging, level, logging.WARNING))
        logger.propagate = False
    return logger
