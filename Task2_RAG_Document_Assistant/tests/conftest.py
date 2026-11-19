"""Shared fixtures. Tests use the offline test backends (hash embeddings + mock LLM)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Settings  # noqa: E402
from src.document_loader import load_document  # noqa: E402
from src.embeddings import get_embeddings  # noqa: E402
from src.text_splitter import split_documents  # noqa: E402
from src.vector_store import VectorStoreManager  # noqa: E402
from tests.make_fixtures import build_fixtures  # noqa: E402


@pytest.fixture(scope="session")
def fixture_paths(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Freshly generated sample documents (PDF, TXT, empty, blank, corrupted)."""
    return build_fixtures(tmp_path_factory.mktemp("fixtures"))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Offline test settings pointing at throw-away directories."""
    return Settings(
        llm_provider="mock",
        embedding_backend="hash",
        embedding_model="hashing-test",
        chunk_size=400,
        chunk_overlap=50,
        top_k=4,
        relevance_threshold=0.2,
        chroma_dir=tmp_path / "chroma",
        upload_dir=tmp_path / "uploads",
    )


@pytest.fixture
def store(settings: Settings) -> VectorStoreManager:
    return VectorStoreManager(
        persist_dir=settings.chroma_dir,
        embeddings=get_embeddings(settings),
        collection_name=settings.collection_name,
        embedding_id=settings.embedding_id,
    )


@pytest.fixture
def indexed_store(store: VectorStoreManager, settings: Settings, fixture_paths: dict[str, Path]) -> VectorStoreManager:
    """A store containing both the ML PDF and the solar TXT document."""
    for key in ("ml_pdf", "solar_txt"):
        pages = load_document(fixture_paths[key])
        store.add_chunks(split_documents(pages, settings.chunk_size, settings.chunk_overlap))
    return store
