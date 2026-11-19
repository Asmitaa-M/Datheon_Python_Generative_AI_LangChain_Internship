"""Small, dependency-free helpers: logging, file I/O and input checks."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from exceptions import EmptyInputError, InputFileError, OutputFileError

PathLike = Union[str, "os.PathLike[str]"]

REDACTED = "[REDACTED]"

# Google API keys start with "AIza"; the second pattern catches "key=..." forms.
_GOOGLE_KEY_RE = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")
_KEY_ASSIGNMENT_RE = re.compile(
    r"""(?ix)\b([\w-]*key\s*[=:]\s*)([^\s,;&"']+)"""
)


# --------------------------------------------------------------------------- #
# Secrets and logging
# --------------------------------------------------------------------------- #
def redact_secrets(text: str, secrets: Iterable[Optional[str]] = ()) -> str:
    """Remove API keys from ``text`` so it is safe to log or display.

    Args:
        text: The text to clean.
        secrets: Exact secret values (for example the configured API key) that
            must never appear in the output.
    """
    for secret in secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, REDACTED)
    text = _GOOGLE_KEY_RE.sub(REDACTED, text)
    return _KEY_ASSIGNMENT_RE.sub(lambda m: m.group(1) + REDACTED, text)


class RedactingFormatter(logging.Formatter):
    """Log formatter that scrubs secrets from the whole output, tracebacks too."""

    def __init__(self, fmt: str, secrets: Iterable[Optional[str]] = ()) -> None:
        super().__init__(fmt)
        self._secrets = [s for s in secrets if s]

    def format(self, record: logging.LogRecord) -> str:
        return redact_secrets(super().format(record), self._secrets)


_active_handler: Optional[logging.Handler] = None


def setup_logging(
    verbose: bool = False, secrets: Iterable[Optional[str]] = ()
) -> None:
    """Configure console logging (stderr) with secret redaction.

    Logs go to stderr so that stdout only carries the extracted JSON. Calling
    this more than once replaces the previous handler instead of duplicating it.
    """
    global _active_handler
    root = logging.getLogger()
    if _active_handler is not None:
        root.removeHandler(_active_handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        RedactingFormatter("%(levelname)s %(name)s: %(message)s", secrets)
    )
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    _active_handler = handler

    # Third-party HTTP libraries are chatty; keep them quiet.
    for noisy in ("httpx", "httpx2", "httpcore", "google_genai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #
def ensure_non_empty_text(text: Optional[str]) -> str:
    """Return ``text`` stripped of outer whitespace.

    Raises:
        EmptyInputError: If the text is ``None``, empty, or only whitespace.
    """
    if text is None or not text.strip():
        raise EmptyInputError(
            "The document text is empty. Provide some text to extract from."
        )
    return text.strip()


def read_text_file(path: PathLike) -> str:
    """Read a UTF-8 text file (a leading BOM from Windows Notepad is accepted).

    Raises:
        InputFileError: If the file is missing, unreadable, or not UTF-8.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise InputFileError(f"Input file not found: {file_path}")
    if file_path.is_dir():
        raise InputFileError(f"Input path is a directory, not a file: {file_path}")
    try:
        return file_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputFileError(
            f"Input file is not valid UTF-8 text: {file_path}. "
            "Re-save it with UTF-8 encoding."
        ) from exc
    except OSError as exc:
        raise InputFileError(f"Cannot read input file {file_path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def write_json_file(data: Any, path: PathLike) -> Path:
    """Write ``data`` as pretty-printed UTF-8 JSON and return the path.

    The file is written to a temporary file first and then moved into place, so
    a crash can never leave a half-written result behind.

    Raises:
        OutputFileError: If the data cannot be serialised or the file cannot
            be written.
    """
    target = Path(path)
    try:
        payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise OutputFileError(f"Result cannot be converted to JSON: {exc}") from exc

    temp_path: Optional[Path] = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        os.replace(temp_path, target)
        temp_path = None
    except OSError as exc:
        raise OutputFileError(f"Cannot write output file {target}: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return target


def build_output_path(
    output_dir: PathLike,
    document_type: str,
    source_name: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Path:
    """Build a unique, filesystem-safe output path.

    Example: ``output/invoice_sample_invoice_20261116_001234.json``
    """
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    parts = [document_type]
    if source_name:
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(source_name).stem).strip("_")
        if stem:
            parts.append(stem)
    parts.append(timestamp)

    directory = Path(output_dir)
    base = "_".join(parts)
    candidate = directory / f"{base}.json"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{base}_{counter}.json"
        counter += 1
    return candidate
