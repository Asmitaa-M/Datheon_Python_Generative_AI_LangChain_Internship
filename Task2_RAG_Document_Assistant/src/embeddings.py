"""Embedding backends.

* ``huggingface`` (default, production): a HuggingFace sentence-transformers
  model such as ``all-MiniLM-L6-v2``, loaded through ``langchain-huggingface``.
* ``hash`` (offline TEST MODE ONLY): deterministic bag-of-words hashing vectors.
  It needs no model download and is used by the automated tests where
  Hugging Face is unreachable. It captures lexical overlap, not semantics, and
  must not be used for real question answering.

The same ``Embeddings`` object is used for indexing and for queries, so
documents and questions always live in the same vector space.
"""

from __future__ import annotations

import hashlib
import math

from langchain_core.embeddings import Embeddings

from config import Settings
from src.utils import tokenize_content


class EmbeddingError(RuntimeError):
    """Raised when an embedding model cannot be loaded or fails to embed text."""


class HashingEmbeddings(Embeddings):
    """Deterministic, dependency-free embeddings for offline testing."""

    def __init__(self, dimensions: int = 512) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than 0.")
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize_content(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class GuardedEmbeddings(Embeddings):
    """Wrap any ``Embeddings`` so every failure surfaces as ``EmbeddingError``."""

    def __init__(self, inner: Embeddings, description: str) -> None:
        self._inner = inner
        self.description = description

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = self._inner.embed_documents(texts)
        except Exception as exc:
            raise EmbeddingError(f"Embedding documents with {self.description} failed: {exc}") from exc
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise EmbeddingError(f"{self.description} returned an invalid number of embeddings.")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed an empty query.")
        try:
            vector = self._inner.embed_query(text)
        except Exception as exc:
            raise EmbeddingError(f"Embedding the query with {self.description} failed: {exc}") from exc
        if not vector:
            raise EmbeddingError(f"{self.description} returned an empty query embedding.")
        return vector


def get_embeddings(settings: Settings) -> Embeddings:
    """Create the embedding model configured in ``settings``.

    Raises:
        EmbeddingError: if the model cannot be loaded (e.g. no network access
            to download it for the first time).
    """
    if settings.embedding_backend == "hash":
        return GuardedEmbeddings(HashingEmbeddings(), "hashing test embeddings")

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        inner = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise EmbeddingError(
            f"Could not load embedding model '{settings.embedding_model}': {exc}. "
            "The first run downloads the model from Hugging Face, so internet access is "
            "required once. You can also point EMBEDDING_MODEL at a local model folder."
        ) from exc
    return GuardedEmbeddings(inner, f"HuggingFace model '{settings.embedding_model}'")
