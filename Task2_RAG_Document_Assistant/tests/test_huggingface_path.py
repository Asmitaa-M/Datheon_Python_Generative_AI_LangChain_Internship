"""Exercise the production HuggingFace/sentence-transformers code path offline.

The real ``all-MiniLM-L6-v2`` weights need a one-off download from Hugging Face.
To prove the *integration* (langchain-huggingface -> sentence-transformers ->
Chroma) without network access, these tests build a tiny, randomly initialised
sentence-transformers model on disk and point ``EMBEDDING_MODEL`` at it.

IMPORTANT: random weights carry no semantic meaning, so these tests check that
the pipeline runs and persists correctly - NOT retrieval quality.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("sentence_transformers")

from config import Settings  # noqa: E402
from src.document_loader import load_document  # noqa: E402
from src.embeddings import get_embeddings  # noqa: E402
from src.text_splitter import split_documents  # noqa: E402
from src.vector_store import VectorStoreManager  # noqa: E402

DIMENSIONS = 32


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import models as st_models
    from tokenizers import Tokenizer
    from tokenizers import models as tok_models
    from tokenizers import pre_tokenizers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    root = tmp_path_factory.mktemp("tiny_model")
    hf_dir, st_dir = root / "hf", root / "st"
    hf_dir.mkdir()

    words = "[PAD] [UNK] [CLS] [SEP] machine learning solar energy panel data model".split()
    tokenizer = Tokenizer(tok_models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]"
    )
    config = BertConfig(
        vocab_size=len(words),
        hidden_size=DIMENSIONS,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    BertModel(config).save_pretrained(hf_dir)
    fast.save_pretrained(hf_dir)

    transformer = st_models.Transformer(str(hf_dir), max_seq_length=32)
    pooling = st_models.Pooling(transformer.get_embedding_dimension(), "mean")
    SentenceTransformer(modules=[transformer, pooling]).save(str(st_dir))
    return st_dir


@pytest.fixture
def hf_settings(tiny_model_dir: Path, tmp_path: Path) -> Settings:
    return Settings(
        llm_provider="mock",
        embedding_backend="huggingface",
        embedding_model=str(tiny_model_dir),
        chunk_size=400,
        chunk_overlap=50,
        chroma_dir=tmp_path / "chroma",
        upload_dir=tmp_path / "uploads",
    )


def test_huggingface_backend_loads_a_sentence_transformer_model(hf_settings: Settings) -> None:
    model = get_embeddings(hf_settings)
    vectors = model.embed_documents(["machine learning data", "solar energy panel"])
    query = model.embed_query("machine learning data")
    assert len(vectors) == 2 and all(len(v) == DIMENSIONS for v in vectors)
    for vector in [*vectors, query]:
        assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-4)
    # Same model for documents and queries -> identical text gives identical vectors.
    assert all(math.isclose(a, b, abs_tol=1e-5) for a, b in zip(query, vectors[0]))


def test_huggingface_vectors_flow_through_persistent_chroma(hf_settings: Settings, fixture_paths: dict[str, Path]) -> None:
    def open_store() -> VectorStoreManager:
        return VectorStoreManager(
            hf_settings.chroma_dir, get_embeddings(hf_settings), hf_settings.collection_name, hf_settings.embedding_id
        )

    store = open_store()
    chunks = split_documents(load_document(fixture_paths["ml_pdf"]), 400, 50)
    assert store.add_chunks(chunks) == len(chunks)
    del store

    reopened = open_store()
    assert reopened.count() == len(chunks)
    results = reopened.similarity_search_with_relevance("machine learning", k=3)
    assert len(results) == 3
    for document, score in results:
        assert -1.0 <= score <= 1.0
        assert document.metadata["source"] == "machine_learning.pdf"
        assert document.metadata["page_number"] in (1, 2, 3)
