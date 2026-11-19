"""ChromaDB persistence, de-duplication and clearing."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from config import Settings
from src.document_loader import load_document
from src.embeddings import get_embeddings
from src.text_splitter import split_documents
from src.vector_store import VectorStoreError, VectorStoreManager

ROOT = Path(__file__).resolve().parents[1]


def _chunks(path: Path, settings: Settings):
    return split_documents(load_document(path), settings.chunk_size, settings.chunk_overlap)


def _open(settings: Settings, embedding_id: str | None = None) -> VectorStoreManager:
    return VectorStoreManager(
        persist_dir=settings.chroma_dir,
        embeddings=get_embeddings(settings),
        collection_name=settings.collection_name,
        embedding_id=embedding_id or settings.embedding_id,
    )


def test_add_and_count(store: VectorStoreManager, settings: Settings, fixture_paths: dict[str, Path]) -> None:
    chunks = _chunks(fixture_paths["ml_pdf"], settings)
    assert store.add_chunks(chunks) == len(chunks)
    assert store.count() == len(chunks)


def test_adding_same_chunks_twice_does_not_duplicate(
    store: VectorStoreManager, settings: Settings, fixture_paths: dict[str, Path]
) -> None:
    chunks = _chunks(fixture_paths["ml_pdf"], settings)
    store.add_chunks(chunks)
    assert store.add_chunks(chunks) == 0
    assert store.count() == len(chunks)


def test_has_document_uses_content_hash(
    store: VectorStoreManager, settings: Settings, fixture_paths: dict[str, Path]
) -> None:
    chunks = _chunks(fixture_paths["ml_pdf"], settings)
    doc_hash = chunks[0].metadata["doc_hash"]
    assert not store.has_document(doc_hash)
    store.add_chunks(chunks)
    assert store.has_document(doc_hash)
    assert not store.has_document("0" * 64)


def test_list_documents_summarises_each_file(indexed_store: VectorStoreManager) -> None:
    documents = indexed_store.list_documents()
    assert [doc.source for doc in documents] == ["machine_learning.pdf", "solar_energy.txt"]
    ml, solar = documents
    assert ml.file_type == "pdf" and ml.page_count == 3
    assert solar.file_type == "txt" and solar.chunk_count >= 1


def test_chunk_metadata_round_trips_through_chroma(indexed_store: VectorStoreManager) -> None:
    results = indexed_store.similarity_search_with_relevance("What is overfitting?", k=1)
    document, score = results[0]
    assert document.metadata["source"] == "machine_learning.pdf"
    assert document.metadata["page_number"] == 3
    assert document.metadata["chunk_id"]
    assert 0.0 < score <= 1.0


def test_data_persists_across_new_manager_instances(
    settings: Settings, fixture_paths: dict[str, Path]
) -> None:
    first = _open(settings)
    chunks = _chunks(fixture_paths["ml_pdf"], settings)
    first.add_chunks(chunks)
    del first

    second = _open(settings)
    assert second.count() == len(chunks)
    top = second.similarity_search_with_relevance("What is machine learning?", k=1)[0][0]
    assert top.metadata["source"] == "machine_learning.pdf"


def test_data_persists_across_a_real_process_restart(
    settings: Settings, fixture_paths: dict[str, Path]
) -> None:
    """Write in this process, read back from a brand-new Python process."""
    writer = _open(settings)
    chunks = _chunks(fixture_paths["ml_pdf"], settings)
    writer.add_chunks(chunks)
    expected = len(chunks)
    del writer

    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        from pathlib import Path
        from config import Settings
        from src.embeddings import get_embeddings
        from src.vector_store import VectorStoreManager

        s = Settings(embedding_backend="hash", embedding_model="hashing-test", llm_provider="mock",
                     chroma_dir=Path({str(settings.chroma_dir)!r}))
        store = VectorStoreManager(s.chroma_dir, get_embeddings(s), s.collection_name, s.embedding_id)
        top = store.similarity_search_with_relevance("What is overfitting?", k=1)[0][0]
        print("COUNT", store.count(), "SOURCE", top.metadata["source"], "PAGE", top.metadata["page_number"])
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert f"COUNT {expected} SOURCE machine_learning.pdf PAGE 3" in completed.stdout


def test_clear_removes_everything_and_store_stays_usable(
    indexed_store: VectorStoreManager, settings: Settings, fixture_paths: dict[str, Path]
) -> None:
    assert indexed_store.count() > 0
    indexed_store.clear()
    assert indexed_store.count() == 0
    assert indexed_store.list_documents() == []
    assert indexed_store.similarity_search_with_relevance("machine learning") == []
    chunks = _chunks(fixture_paths["solar_txt"], settings)
    assert indexed_store.add_chunks(chunks) == len(chunks)


def test_clear_persists_after_reopen(indexed_store: VectorStoreManager, settings: Settings) -> None:
    indexed_store.clear()
    del indexed_store
    assert _open(settings).count() == 0


def test_search_on_empty_store_returns_nothing(store: VectorStoreManager) -> None:
    assert store.similarity_search_with_relevance("anything") == []


def test_mixing_embedding_models_is_refused(store: VectorStoreManager, settings: Settings) -> None:
    del store
    with pytest.raises(VectorStoreError, match="built with embedding model"):
        _open(settings, embedding_id="huggingface:some-other-model")
