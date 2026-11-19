"""Relevant retrieval, thresholding and correct page metadata."""

from __future__ import annotations

from src.retriever import ThresholdRetriever
from src.vector_store import VectorStoreManager


def _retriever(store: VectorStoreManager, k: int = 4, threshold: float = 0.2) -> ThresholdRetriever:
    return ThresholdRetriever(store=store, k=k, score_threshold=threshold)


def test_relevant_chunk_is_retrieved_from_correct_document_and_page(indexed_store: VectorStoreManager) -> None:
    result = _retriever(indexed_store).retrieve("What is overfitting and how is it reduced?")
    assert result.accepted
    top = result.accepted[0]
    assert top.metadata["source"] == "machine_learning.pdf"
    assert top.metadata["page_number"] == 3
    assert "Overfitting" in top.page_content


def test_definition_question_cites_page_one(indexed_store: VectorStoreManager) -> None:
    top = _retriever(indexed_store).retrieve("What is machine learning?").accepted[0]
    assert top.metadata["source"] == "machine_learning.pdf"
    assert top.metadata["page_number"] == 1


def test_txt_document_is_retrievable(indexed_store: VectorStoreManager) -> None:
    top = _retriever(indexed_store).retrieve("How do solar panels convert sunlight into electricity?").accepted[0]
    assert top.metadata["source"] == "solar_energy.txt"


def test_accepted_chunks_carry_their_score(indexed_store: VectorStoreManager) -> None:
    result = _retriever(indexed_store).retrieve("What is supervised learning?")
    assert all(0.2 <= doc.metadata["score"] <= 1.0 for doc in result.accepted)
    scores = [doc.metadata["score"] for doc in result.accepted]
    assert scores == sorted(scores, reverse=True)


def test_irrelevant_question_is_rejected_by_threshold(indexed_store: VectorStoreManager) -> None:
    result = _retriever(indexed_store).retrieve("Who won the football world cup in 2018?")
    assert result.accepted == []
    assert result.rejected_count > 0
    assert result.best_score is not None and result.best_score < 0.2


def test_threshold_is_configurable(indexed_store: VectorStoreManager) -> None:
    question = "What is machine learning?"
    lenient = _retriever(indexed_store, threshold=-1.0).retrieve(question)
    strict = _retriever(indexed_store, threshold=0.99).retrieve(question)
    assert len(lenient.accepted) > len(strict.accepted)
    assert strict.accepted == []


def test_top_k_limits_results(indexed_store: VectorStoreManager) -> None:
    result = _retriever(indexed_store, k=2, threshold=-1.0).retrieve("learning")
    assert len(result.accepted) == 2


def test_works_through_the_standard_langchain_invoke_api(indexed_store: VectorStoreManager) -> None:
    docs = _retriever(indexed_store).invoke("What is reinforcement learning?")
    assert docs and docs[0].metadata["source"] == "machine_learning.pdf"
    assert docs[0].metadata["page_number"] == 2


def test_empty_store_returns_nothing(store: VectorStoreManager) -> None:
    result = _retriever(store).retrieve("What is machine learning?")
    assert result.accepted == [] and result.best_score is None
