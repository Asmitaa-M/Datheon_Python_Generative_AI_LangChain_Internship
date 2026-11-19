"""Recursive chunking that preserves metadata and assigns stable chunk IDs."""

from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into overlapping chunks.

    Every chunk keeps its parent's metadata (``source``, ``page_number``,
    ``file_type``, ``doc_hash``) and gains:

    * ``chunk_id`` - deterministic ID ``<hash16>-p<page>-c<n>``; re-indexing the
      same file produces the same IDs, which is what prevents duplicates.
    * ``chunk_index`` - position of the chunk within the whole document.
    * ``start_index`` - character offset inside the parent page.

    Raises:
        ValueError: if ``chunk_size``/``chunk_overlap`` are inconsistent.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    per_page_counter: dict[tuple[str, int], int] = defaultdict(int)
    per_doc_counter: dict[str, int] = defaultdict(int)
    chunks: list[Document] = []

    for chunk in splitter.split_documents(documents):
        text = chunk.page_content.strip()
        if not text:
            continue
        doc_hash = str(chunk.metadata.get("doc_hash", "nohash"))
        page = int(chunk.metadata.get("page_number", 1))

        page_position = per_page_counter[(doc_hash, page)]
        per_page_counter[(doc_hash, page)] += 1
        doc_position = per_doc_counter[doc_hash]
        per_doc_counter[doc_hash] += 1

        metadata = dict(chunk.metadata)
        metadata["chunk_id"] = f"{doc_hash[:16]}-p{page}-c{page_position}"
        metadata["chunk_index"] = doc_position
        chunks.append(Document(page_content=text, metadata=metadata))

    return chunks
