"""Persistent ChromaDB vector store."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.embeddings import EmbeddingError

INDEX_INFO_FILE = "index_info.json"
ADD_BATCH_SIZE = 128


class VectorStoreError(RuntimeError):
    """Raised for vector-store problems (corruption, model mismatch, ...)."""


@dataclass(frozen=True)
class IndexedDocument:
    """Summary of one indexed file."""

    source: str
    doc_hash: str
    file_type: str
    chunk_count: int
    page_count: int


class VectorStoreManager:
    """Create, persist, query and clear the Chroma collection.

    Embeddings are stored with cosine distance, so ``1 - distance`` is the
    cosine similarity returned by :meth:`similarity_search_with_relevance`.
    """

    def __init__(
        self,
        persist_dir: Path,
        embeddings: Embeddings,
        collection_name: str = "documents",
        embedding_id: str = "unknown",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.embedding_id = embedding_id
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._check_embedding_compatibility()
            self._chroma = self._create_chroma()
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Could not open the vector database at {self.persist_dir}: {exc}") from exc

    # -- internals -------------------------------------------------------
    def _create_chroma(self) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
            client_settings=ChromaSettings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory=str(self.persist_dir),
            ),
        )

    @property
    def _info_path(self) -> Path:
        return self.persist_dir / INDEX_INFO_FILE

    def _check_embedding_compatibility(self) -> None:
        """Refuse to mix vectors made by different embedding models."""
        if not self._info_path.exists():
            self._info_path.write_text(json.dumps({"embedding_id": self.embedding_id}), encoding="utf-8")
            return
        try:
            stored = json.loads(self._info_path.read_text(encoding="utf-8")).get("embedding_id")
        except (OSError, json.JSONDecodeError) as exc:
            raise VectorStoreError(f"Index metadata file is unreadable: {exc}") from exc
        if stored != self.embedding_id:
            raise VectorStoreError(
                f"The existing index was built with embedding model '{stored}' but the app is "
                f"configured for '{self.embedding_id}'. Clear the database (or restore the old "
                "EMBEDDING_MODEL) - vectors from different models cannot be compared."
            )

    # -- queries ---------------------------------------------------------
    def count(self) -> int:
        """Number of chunks currently stored."""
        try:
            return int(self._chroma._collection.count())
        except Exception as exc:
            raise VectorStoreError(f"Could not count stored chunks: {exc}") from exc

    def has_document(self, doc_hash: str) -> bool:
        """True if a file with this content hash is already indexed."""
        try:
            found = self._chroma.get(where={"doc_hash": doc_hash}, limit=1, include=[])
        except Exception as exc:
            raise VectorStoreError(f"Could not check for an existing document: {exc}") from exc
        return bool(found["ids"])

    def list_documents(self) -> list[IndexedDocument]:
        """Summaries of every indexed file, sorted by filename."""
        try:
            data = self._chroma.get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"Could not list indexed documents: {exc}") from exc

        chunk_counts: dict[str, int] = defaultdict(int)
        pages: dict[str, set[int]] = defaultdict(set)
        info: dict[str, tuple[str, str]] = {}
        for metadata in data["metadatas"] or []:
            doc_hash = str(metadata.get("doc_hash", ""))
            chunk_counts[doc_hash] += 1
            pages[doc_hash].add(int(metadata.get("page_number", 1)))
            info[doc_hash] = (str(metadata.get("source", "unknown")), str(metadata.get("file_type", "")))

        documents = [
            IndexedDocument(
                source=info[doc_hash][0],
                doc_hash=doc_hash,
                file_type=info[doc_hash][1],
                chunk_count=chunk_counts[doc_hash],
                page_count=len(pages[doc_hash]),
            )
            for doc_hash in info
        ]
        return sorted(documents, key=lambda doc: doc.source.lower())

    def similarity_search_with_relevance(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Return up to ``k`` ``(chunk, cosine_similarity)`` pairs, best first."""
        if self.count() == 0:
            return []
        try:
            results = self._chroma.similarity_search_with_score(query, k=k)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Similarity search failed: {exc}") from exc
        return [(document, 1.0 - float(distance)) for document, distance in results]

    # -- writes ----------------------------------------------------------
    def add_chunks(self, chunks: Sequence[Document]) -> int:
        """Add chunks, skipping any whose ``chunk_id`` is already stored.

        Returns:
            Number of chunks actually written.
        """
        if not chunks:
            return 0
        ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
        try:
            existing = set(self._chroma.get(ids=ids, include=[])["ids"])
        except Exception as exc:
            raise VectorStoreError(f"Could not check existing chunks: {exc}") from exc

        new_chunks: list[Document] = []
        new_ids: list[str] = []
        for chunk_id, chunk in zip(ids, chunks):
            if chunk_id in existing or chunk_id in new_ids:
                continue
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

        written = 0
        for start in range(0, len(new_chunks), ADD_BATCH_SIZE):
            batch = new_chunks[start : start + ADD_BATCH_SIZE]
            batch_ids = new_ids[start : start + ADD_BATCH_SIZE]
            try:
                self._chroma.add_documents(batch, ids=batch_ids)
            except EmbeddingError:
                raise
            except Exception as exc:
                raise VectorStoreError(f"Could not store chunks: {exc}") from exc
            written += len(batch)
        return written

    def clear(self) -> None:
        """Delete every indexed chunk and reset the index metadata."""
        try:
            self._chroma.delete_collection()
            self._info_path.unlink(missing_ok=True)
            self._info_path.write_text(json.dumps({"embedding_id": self.embedding_id}), encoding="utf-8")
            self._chroma = self._create_chroma()
        except Exception as exc:
            raise VectorStoreError(f"Could not clear the vector database: {exc}") from exc
