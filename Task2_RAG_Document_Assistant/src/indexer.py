"""Upload -> validate -> load -> split -> embed -> store pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from config import Settings
from src.document_loader import DocumentLoadError, load_document
from src.embeddings import EmbeddingError
from src.text_splitter import split_documents
from src.utils import (
    ValidationError,
    get_extension,
    sanitize_filename,
    sha256_bytes,
    validate_extension,
    validate_size,
)
from src.vector_store import VectorStoreError, VectorStoreManager

ProgressCallback = Callable[[float, str], None]

INDEXED = "indexed"
DUPLICATE = "duplicate"
FAILED = "failed"


@dataclass(frozen=True)
class UploadedItem:
    """A file received from the user (name + raw bytes)."""

    name: str
    data: bytes


@dataclass(frozen=True)
class IndexResult:
    """Outcome of indexing a single file."""

    filename: str
    status: str  # indexed | duplicate | failed
    message: str
    chunks: int = 0


def index_uploads(
    items: Sequence[UploadedItem],
    store: VectorStoreManager,
    settings: Settings,
    progress: ProgressCallback | None = None,
) -> list[IndexResult]:
    """Index each uploaded file, reporting a per-file result.

    One bad file never aborts the others. Existing indexed documents are kept,
    and a file whose contents are already indexed is skipped as a duplicate.
    """
    results: list[IndexResult] = []
    total = max(len(items), 1)
    for position, item in enumerate(items):
        if progress:
            progress(position / total, f"Processing {item.name} ({position + 1}/{len(items)})")
        results.append(_index_one(item, store, settings))
    if progress:
        progress(1.0, "Done")
    return results


def _index_one(item: UploadedItem, store: VectorStoreManager, settings: Settings) -> IndexResult:
    display_name = sanitize_filename(item.name)
    saved_path: Path | None = None
    try:
        validate_extension(display_name)
        validate_size(display_name, len(item.data), settings.max_upload_mb)

        doc_hash = sha256_bytes(item.data)
        if store.has_document(doc_hash):
            return IndexResult(display_name, DUPLICATE, "Already indexed - skipped.")

        saved_path = _save_upload(item.data, display_name, doc_hash, settings.upload_dir)
        pages = load_document(saved_path, display_name=display_name)
        chunks = split_documents(pages, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise DocumentLoadError(f"'{display_name}' produced no text chunks.")
        written = store.add_chunks(chunks)
        return IndexResult(display_name, INDEXED, f"Indexed {written} chunks from {len(pages)} page(s).", written)
    except (ValidationError, DocumentLoadError, EmbeddingError, VectorStoreError) as exc:
        if saved_path is not None:
            saved_path.unlink(missing_ok=True)
        return IndexResult(display_name, FAILED, str(exc))


def _save_upload(data: bytes, display_name: str, doc_hash: str, upload_dir: Path) -> Path:
    """Write the upload to disk, avoiding clashes between different files with one name."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / display_name
    if target.exists() and sha256_bytes(target.read_bytes()) != doc_hash:
        target = upload_dir / f"{target.stem}-{doc_hash[:8]}{get_extension(display_name)}"
    target.write_bytes(data)
    return target
