"""Read-only local file reader confined to a workspace directory."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from core.exceptions import FileAccessError, ToolInputError
from core.models import Source, ToolResult
from core.validators import validate_text

ALLOWED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".py"}


class ReadFileInput(BaseModel):
    path: str = Field(
        ...,
        min_length=1,
        max_length=260,
        description="Path of the file relative to the workspace, e.g. 'expenses.csv'.",
    )


class FileReader:
    """Reads (never writes, deletes or executes) supported files inside ``workspace``."""

    def __init__(self, workspace: Path, max_bytes: int = 100_000):
        self.root = Path(workspace).resolve()
        self.max_bytes = max_bytes

    # -------------------------------------------------------------- helpers
    def _display(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.name

    def available_files(self, limit: int = 20) -> list[str]:
        if not self.root.is_dir():
            return []
        found = [
            p for p in self.root.rglob("*")
            if p.suffix.lower() in ALLOWED_EXTENSIONS and p.is_file()
        ]
        return sorted(self._display(p.resolve()) for p in found)[:limit]

    def resolve(self, path: str) -> Path:
        """Map ``path`` to a real file path inside the workspace or raise."""
        try:
            raw = validate_text(path, name="path", max_len=260)
        except ToolInputError as exc:
            raise FileAccessError(str(exc)) from exc
        raw = raw.strip("\"'")
        if os.sep == "/":                       # tolerate Windows-style separators
            raw = raw.replace("\\", "/")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()          # collapses '..' and follows symlinks
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise FileAccessError(
                "Access denied: only files inside the workspace directory can be read."
            ) from None
        return resolved

    # ----------------------------------------------------------------- read
    def read(self, path: str) -> ToolResult:
        try:
            return self._read(path)
        except FileAccessError as exc:
            return ToolResult.failure("read_file", str(exc))

    def _read(self, path: str) -> ToolResult:
        resolved = self.resolve(path)
        name = self._display(resolved)

        if resolved.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise FileAccessError(
                f"Unsupported file type '{resolved.suffix or '(none)'}'. "
                f"Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            )
        if not resolved.exists():
            files = self.available_files()
            hint = f" Available files: {', '.join(files)}." if files else " The workspace has no supported files."
            raise FileAccessError(f"File not found: '{name}'.{hint}")
        if not resolved.is_file():
            raise FileAccessError(f"'{name}' is not a regular file.")

        try:
            size = resolved.stat().st_size
            with resolved.open("rb") as fh:
                data = fh.read(self.max_bytes + 1)
        except PermissionError:
            raise FileAccessError(f"Permission denied reading '{name}'.") from None
        except OSError as exc:
            raise FileAccessError(f"Could not read '{name}': {exc.strerror or 'I/O error'}.") from None

        if b"\x00" in data:
            raise FileAccessError(f"'{name}' appears to be a binary file, not text.")

        truncated = len(data) > self.max_bytes
        text = data[: self.max_bytes].decode("utf-8-sig", errors="replace")
        if truncated:
            text = text.rsplit("\n", 1)[0]   # avoid cutting a line (e.g. CSV row) in half

        lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        header = f"File: {name} | Size: {size:,} bytes | Lines shown: {lines}"
        if resolved.suffix.lower() == ".csv":
            header += f" | CSV data rows (excluding header): {max(lines - 1, 0)}"
        if truncated:
            header += f" | TRUNCATED to first {self.max_bytes:,} bytes - data is incomplete"

        return ToolResult.success(
            "read_file",
            content=f"{header}\n\n{text}",
            summary=f"Read {name} ({size:,} bytes, {lines} lines{', truncated' if truncated else ''})",
            sources=[Source(title=name, kind="file")],
            untrusted=True,
            truncated=False,
        )
