"""LangChain retriever with a relevance threshold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


@dataclass
class RetrievalResult:
    """Outcome of one retrieval."""

    accepted: list[Document] = field(default_factory=list)
    rejected_count: int = 0
    best_score: float | None = None


class ThresholdRetriever(BaseRetriever):
    """Return the top-``k`` chunks whose cosine similarity is >= ``score_threshold``.

    Chunks below the threshold are discarded *before* they can reach the LLM,
    which is the first line of defence against answering from irrelevant text.
    Each accepted chunk gets its score in ``metadata["score"]``.

    The threshold is model specific: cosine scores from different embedding
    models are not comparable, so calibrate it for the model you use.
    """

    store: Any
    k: int = 4
    score_threshold: float = 0.25

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve chunks and report how many were rejected."""
        scored = self.store.similarity_search_with_relevance(query, k=self.k)
        result = RetrievalResult(best_score=scored[0][1] if scored else None)
        for document, score in scored:
            if score < self.score_threshold:
                result.rejected_count += 1
                continue
            metadata = dict(document.metadata)
            metadata["score"] = round(float(score), 4)
            result.accepted.append(Document(page_content=document.page_content, metadata=metadata))
        return result

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.retrieve(query).accepted
