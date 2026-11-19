"""Shared helpers: file validation, text cleaning, metadata formatting."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Any, Mapping

ALLOWED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt")

# Minimal English stop-word list, shared by the offline test embeddings and the
# mock LLM so both treat "content words" identically.
STOPWORDS: frozenset[str] = frozenset(
    """a an the and or but if then than so of to in on at by for with about as into from
    is are was were be been being am do does did done have has had having it its this that
    these those there here what which who whom whose when where why how can could should would
    will shall may might must not no nor i you he she we they them his her our their your my me
    us also any some such according document documents text tell explain describe give me
    please""".split()
)


class ValidationError(ValueError):
    """Raised when an uploaded file fails validation."""


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def get_extension(filename: str) -> str:
    """Return the lower-cased extension (including the dot) of ``filename``."""
    return Path(filename).suffix.lower()


def validate_extension(filename: str) -> str:
    """Return the extension if supported, otherwise raise ``ValidationError``."""
    extension = get_extension(filename)
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(ALLOWED_EXTENSIONS)
        raise ValidationError(
            f"'{filename}' has unsupported type {extension or '(none)'}. Allowed types: {allowed}."
        )
    return extension


def validate_size(filename: str, size_bytes: int, max_mb: int) -> None:
    """Reject empty files and files larger than ``max_mb`` megabytes."""
    if size_bytes <= 0:
        raise ValidationError(f"'{filename}' is empty.")
    if size_bytes > max_mb * 1024 * 1024:
        raise ValidationError(f"'{filename}' is larger than the {max_mb} MB upload limit.")


def sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe version of ``filename`` (no directories)."""
    name = Path(filename.replace("\\", "/")).name.strip()
    name = re.sub(r"[^\w.\- ]+", "_", name)
    name = name.strip(" .")
    return name or "document"


def clean_text(text: str) -> str:
    """Normalise whitespace while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "").replace("\x0c", "\n")
    text = re.sub(r"[ \t ​]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize_content(text: str) -> list[str]:
    """Lower-case content words (stop-words removed, crude plural stemming)."""
    tokens: list[str] = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.append(word)
    return tokens


def format_page(metadata: Mapping[str, Any]) -> str:
    """Human-readable page label for a chunk's metadata."""
    if metadata.get("file_type") == "txt":
        return "text file"
    page = metadata.get("page_number")
    return f"page {page}" if page is not None else "page n/a"


def format_citation(metadata: Mapping[str, Any]) -> str:
    """Return e.g. ``report.pdf, page 3``."""
    return f"{metadata.get('source', 'unknown')}, {format_page(metadata)}"


def truncate(text: str, limit: int) -> str:
    """Shorten ``text`` to ``limit`` characters, adding an ellipsis if cut."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def markdown_safe(text: str) -> str:
    """Escape characters that Streamlit's markdown would otherwise interpret."""
    return text.replace("$", "\\$")


def clear_directory(directory: Path) -> int:
    """Delete everything inside ``directory`` (keeping the directory itself)."""
    removed = 0
    if not directory.exists():
        return removed
    for child in directory.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed += 1
    return removed
