"""Load PDF and TXT files into LangChain ``Document`` objects with metadata."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from src.utils import ValidationError, clean_text, sha256_bytes, validate_extension

# Tried in order; latin-1 accepts any byte sequence so it is the last resort.
TEXT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "cp1252", "latin-1")


class DocumentLoadError(Exception):
    """Raised when a document is empty, corrupted, unreadable or unsupported."""


def load_document(path: str | Path, display_name: str | None = None) -> list[Document]:
    """Load one PDF or TXT file.

    Each returned ``Document`` carries this metadata:

    * ``source`` - filename shown to the user
    * ``page_number`` - 1-based page (always 1 for TXT files)
    * ``file_type`` - ``"pdf"`` or ``"txt"``
    * ``doc_hash`` - SHA-256 of the file contents (used for de-duplication)

    Raises:
        DocumentLoadError: if the file is missing, empty, corrupted, encrypted,
            has no extractable text, or has an unsupported type.
    """
    file_path = Path(path)
    name = display_name or file_path.name

    try:
        extension = validate_extension(name)
    except ValidationError as exc:
        raise DocumentLoadError(str(exc)) from exc

    if not file_path.is_file():
        raise DocumentLoadError(f"'{name}' was not found on disk.")
    try:
        raw = file_path.read_bytes()
    except OSError as exc:
        raise DocumentLoadError(f"'{name}' could not be read: {exc}") from exc
    if not raw.strip():
        raise DocumentLoadError(f"'{name}' is empty.")

    doc_hash = sha256_bytes(raw)
    if extension == ".pdf":
        return _load_pdf(file_path, name, doc_hash)
    return _load_txt(file_path, name, doc_hash)


def _load_pdf(path: Path, name: str, doc_hash: str) -> list[Document]:
    try:
        pages = PyPDFLoader(str(path)).load()
    except Exception as exc:  # pypdf raises many unrelated exception types
        raise DocumentLoadError(
            f"'{name}' could not be read as a PDF (it may be corrupted or password-protected): {exc}"
        ) from exc

    documents: list[Document] = []
    for position, page in enumerate(pages):
        text = clean_text(page.page_content)
        if not text:
            continue
        page_index = page.metadata.get("page", position)
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": name,
                    "page_number": int(page_index) + 1,
                    "file_type": "pdf",
                    "doc_hash": doc_hash,
                },
            )
        )

    if not documents:
        raise DocumentLoadError(
            f"'{name}' contains no extractable text (it may be a scanned image-only PDF)."
        )
    return documents


def _load_txt(path: Path, name: str, doc_hash: str) -> list[Document]:
    last_error: Exception | None = None
    loaded: list[Document] = []
    for encoding in TEXT_ENCODINGS:
        try:
            loaded = TextLoader(str(path), encoding=encoding).load()
            break
        except Exception as exc:  # decoding failures surface as several types
            last_error = exc
    else:
        raise DocumentLoadError(f"'{name}' could not be decoded as text: {last_error}")

    text = "\n".join(doc.page_content for doc in loaded)
    if "\x00" in text:
        raise DocumentLoadError(f"'{name}' looks like a binary file, not plain text.")
    text = clean_text(text)
    if not text:
        raise DocumentLoadError(f"'{name}' contains no text.")

    return [
        Document(
            page_content=text,
            metadata={
                "source": name,
                "page_number": 1,
                "file_type": "txt",
                "doc_hash": doc_hash,
            },
        )
    ]
