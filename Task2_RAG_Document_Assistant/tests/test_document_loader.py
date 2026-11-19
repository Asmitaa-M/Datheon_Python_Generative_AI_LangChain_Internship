"""Loading PDFs and TXT files, including empty/corrupted/unreadable inputs."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.document_loader import DocumentLoadError, load_document


def test_pdf_loads_with_page_level_metadata(fixture_paths: dict[str, Path]) -> None:
    pages = load_document(fixture_paths["ml_pdf"])
    assert [doc.metadata["page_number"] for doc in pages] == [1, 2, 3]
    assert all(doc.metadata["source"] == "machine_learning.pdf" for doc in pages)
    assert all(doc.metadata["file_type"] == "pdf" for doc in pages)
    assert "Machine learning is a subfield of artificial intelligence" in pages[0].page_content
    assert "Overfitting" in pages[2].page_content


def test_pdf_metadata_does_not_leak_filesystem_path(fixture_paths: dict[str, Path]) -> None:
    pages = load_document(fixture_paths["ml_pdf"])
    assert all("/" not in doc.metadata["source"] for doc in pages)


def test_txt_loads(fixture_paths: dict[str, Path]) -> None:
    docs = load_document(fixture_paths["solar_txt"])
    assert len(docs) == 1
    metadata = docs[0].metadata
    assert metadata["source"] == "solar_energy.txt"
    assert metadata["file_type"] == "txt"
    assert metadata["page_number"] == 1
    assert "photovoltaic" in docs[0].page_content


def test_display_name_overrides_disk_name(fixture_paths: dict[str, Path]) -> None:
    docs = load_document(fixture_paths["solar_txt"], display_name="renamed.txt")
    assert docs[0].metadata["source"] == "renamed.txt"


def test_whitespace_is_normalised(tmp_path: Path) -> None:
    path = tmp_path / "messy.txt"
    path.write_text("Hello   world \t again\r\n\r\n\r\n\r\nNext   paragraph   here.  \n", encoding="utf-8")
    text = load_document(path)[0].page_content
    assert "  " not in text
    assert "\n\n\n" not in text
    assert text == "Hello world again\n\nNext paragraph here."


def test_doc_hash_is_content_based(tmp_path: Path) -> None:
    first, second, third = tmp_path / "a.txt", tmp_path / "b.txt", tmp_path / "c.txt"
    first.write_text("same content", encoding="utf-8")
    second.write_text("same content", encoding="utf-8")
    third.write_text("different content", encoding="utf-8")
    hash_first = load_document(first)[0].metadata["doc_hash"]
    assert hash_first == load_document(second)[0].metadata["doc_hash"]
    assert hash_first != load_document(third)[0].metadata["doc_hash"]


def test_empty_txt_is_rejected(fixture_paths: dict[str, Path]) -> None:
    with pytest.raises(DocumentLoadError, match="empty"):
        load_document(fixture_paths["empty_txt"])


def test_whitespace_only_txt_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "blank.txt"
    path.write_text("   \n\n\t  \n", encoding="utf-8")
    with pytest.raises(DocumentLoadError):
        load_document(path)


def test_corrupted_pdf_is_rejected(fixture_paths: dict[str, Path]) -> None:
    with pytest.raises(DocumentLoadError, match="could not be read as a PDF"):
        load_document(fixture_paths["corrupted_pdf"])


def test_pdf_without_extractable_text_is_rejected(fixture_paths: dict[str, Path]) -> None:
    with pytest.raises(DocumentLoadError, match="no extractable text"):
        load_document(fixture_paths["blank_pdf"])


def test_binary_file_named_txt_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "binary.txt"
    path.write_bytes(b"\x00\x01\x02\x03binary\x00data\xff\xfe")
    with pytest.raises(DocumentLoadError, match="binary"):
        load_document(path)


def test_latin1_text_is_decoded(tmp_path: Path) -> None:
    path = tmp_path / "latin.txt"
    path.write_bytes("Caf\xe9 society".encode("latin-1"))
    assert "Café society" in load_document(path)[0].page_content


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "notes.docx"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="unsupported type"):
        load_document(path)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError, match="not found"):
        load_document(tmp_path / "ghost.pdf")
