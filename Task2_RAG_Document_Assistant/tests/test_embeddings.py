"""Embedding generation and error handling."""

from __future__ import annotations

import math

import pytest
from langchain_core.embeddings import Embeddings

from config import Settings
from src.embeddings import EmbeddingError, GuardedEmbeddings, HashingEmbeddings, get_embeddings


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_embeddings_have_fixed_dimension_and_unit_norm(settings: Settings) -> None:
    model = get_embeddings(settings)
    vectors = model.embed_documents(["machine learning learns from data", "solar panels make electricity"])
    assert len(vectors) == 2
    assert all(len(vector) == 512 for vector in vectors)
    assert all(math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-6) for vector in vectors)


def test_embeddings_are_deterministic(settings: Settings) -> None:
    model = get_embeddings(settings)
    assert model.embed_query("what is machine learning") == model.embed_query("what is machine learning")


def test_query_and_document_embeddings_share_one_space(settings: Settings) -> None:
    model = get_embeddings(settings)
    text = "supervised learning uses labeled examples"
    assert model.embed_query(text) == model.embed_documents([text])[0]


def test_related_text_scores_higher_than_unrelated(settings: Settings) -> None:
    model = get_embeddings(settings)
    query = model.embed_query("what is machine learning")
    related = model.embed_documents(["machine learning lets computers learn patterns from data"])[0]
    unrelated = model.embed_documents(["solar panels convert sunlight into electricity"])[0]
    assert _cosine(query, related) > _cosine(query, unrelated)


def test_empty_document_list_returns_empty_list(settings: Settings) -> None:
    assert get_embeddings(settings).embed_documents([]) == []


def test_empty_query_raises_embedding_error(settings: Settings) -> None:
    with pytest.raises(EmbeddingError, match="empty query"):
        get_embeddings(settings).embed_query("   ")


class _BrokenModel(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("boom")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("boom")


def test_model_failures_are_wrapped_in_embedding_error() -> None:
    guarded = GuardedEmbeddings(_BrokenModel(), "broken model")
    with pytest.raises(EmbeddingError, match="boom"):
        guarded.embed_documents(["text"])
    with pytest.raises(EmbeddingError, match="boom"):
        guarded.embed_query("text")


def test_invalid_dimensions_rejected() -> None:
    with pytest.raises(ValueError):
        HashingEmbeddings(dimensions=0)


def test_huggingface_load_failure_is_reported_clearly(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    bad = Settings(embedding_backend="huggingface", embedding_model=str(tmp_path / "no-such-model"))
    with pytest.raises(EmbeddingError, match="Could not load embedding model"):
        get_embeddings(bad)


@pytest.mark.real_model
def test_real_sentence_transformer_embeds_text() -> None:
    """Runs only where the real model can be loaded (needs Hugging Face access once)."""
    try:
        model = get_embeddings(Settings(embedding_backend="huggingface"))
        vectors = model.embed_documents(["machine learning", "solar energy"])
    except EmbeddingError as exc:
        pytest.skip(f"Real embedding model unavailable in this environment: {exc}")
    assert len(vectors) == 2 and len(vectors[0]) == 384
