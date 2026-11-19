"""Recursive chunking, configurable sizes and metadata preservation."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.document_loader import load_document
from src.text_splitter import split_documents


def _long_document() -> Document:
    paragraphs = [f"Paragraph {i}. " + ("Sentence about topic. " * 12) for i in range(10)]
    return Document(
        page_content="\n\n".join(paragraphs),
        metadata={"source": "long.txt", "page_number": 1, "file_type": "txt", "doc_hash": "a" * 64},
    )


def test_chunks_respect_configured_size() -> None:
    chunks = split_documents([_long_document()], chunk_size=300, chunk_overlap=40)
    assert len(chunks) > 3
    assert all(len(chunk.page_content) <= 300 for chunk in chunks)


def test_smaller_chunk_size_creates_more_chunks() -> None:
    doc = _long_document()
    assert len(split_documents([doc], 200, 20)) > len(split_documents([doc], 800, 20))


def test_overlap_is_applied() -> None:
    doc = Document(
        page_content=" ".join(f"word{i}" for i in range(200)),
        metadata={"source": "x.txt", "page_number": 1, "file_type": "txt", "doc_hash": "b" * 64},
    )
    chunks = split_documents([doc], chunk_size=100, chunk_overlap=30)
    first_words = set(chunks[0].page_content.split())
    second_words = set(chunks[1].page_content.split())
    assert first_words & second_words, "consecutive chunks should share overlapping text"


def test_recursive_splitting_prefers_paragraph_boundaries() -> None:
    doc = Document(
        page_content="First paragraph is here.\n\nSecond paragraph is here.\n\nThird paragraph is here.",
        metadata={"source": "p.txt", "page_number": 1, "file_type": "txt", "doc_hash": "c" * 64},
    )
    chunks = split_documents([doc], chunk_size=30, chunk_overlap=0)
    assert [chunk.page_content for chunk in chunks] == [
        "First paragraph is here.",
        "Second paragraph is here.",
        "Third paragraph is here.",
    ]


def test_metadata_is_preserved_and_chunk_ids_added(fixture_paths: dict[str, Path]) -> None:
    pages = load_document(fixture_paths["ml_pdf"])
    chunks = split_documents(pages, chunk_size=200, chunk_overlap=20)
    assert chunks
    for chunk in chunks:
        assert chunk.metadata["source"] == "machine_learning.pdf"
        assert chunk.metadata["file_type"] == "pdf"
        assert chunk.metadata["page_number"] in (1, 2, 3)
        assert chunk.metadata["doc_hash"]
        assert chunk.metadata["chunk_id"].startswith(chunk.metadata["doc_hash"][:16])
        assert isinstance(chunk.metadata["chunk_index"], int)


def test_chunk_page_numbers_match_source_page(fixture_paths: dict[str, Path]) -> None:
    pages = load_document(fixture_paths["ml_pdf"])
    chunks = split_documents(pages, chunk_size=200, chunk_overlap=20)
    for chunk in chunks:
        if "Overfitting" in chunk.page_content:
            assert chunk.metadata["page_number"] == 3
        if "Supervised learning" in chunk.page_content:
            assert chunk.metadata["page_number"] == 2


def test_chunk_ids_are_unique_and_deterministic(fixture_paths: dict[str, Path]) -> None:
    pages = load_document(fixture_paths["ml_pdf"])
    first = [c.metadata["chunk_id"] for c in split_documents(pages, 200, 20)]
    second = [c.metadata["chunk_id"] for c in split_documents(pages, 200, 20)]
    assert len(first) == len(set(first))
    assert first == second


def test_chunk_index_increases_through_document() -> None:
    chunks = split_documents([_long_document()], 300, 40)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


@pytest.mark.parametrize("size, overlap", [(0, 0), (-5, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_configuration_raises(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_documents([_long_document()], size, overlap)
