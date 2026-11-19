"""Headless UI tests using Streamlit's AppTest (no browser needed).

File upload widgets are not drivable through AppTest, so documents are indexed
with the same indexer function the upload button calls, and the UI is then
exercised for chat, citations, clearing and persistence.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from config import Settings
from src.embeddings import get_embeddings
from src.indexer import UploadedItem, index_uploads
from src.qa_chain import REFUSAL_MESSAGE
from src.vector_store import VectorStoreManager

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(ROOT / "app.py")
TITLE = "Document Intelligence – RAG Assistant"


@pytest.fixture(autouse=True)
def offline_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    """Point the app at throw-away storage and the offline test backends."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("EMBEDDING_MODEL", "hashing-test")
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.2")
    monkeypatch.setenv("CHUNK_SIZE", "400")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    st.cache_resource.clear()
    yield Settings.from_env()
    st.cache_resource.clear()


def _index(settings: Settings, fixture_paths: dict[str, Path], *keys: str) -> None:
    store = VectorStoreManager(
        settings.chroma_dir, get_embeddings(settings), settings.collection_name, settings.embedding_id
    )
    items = [UploadedItem(fixture_paths[k].name, fixture_paths[k].read_bytes()) for k in keys]
    index_uploads(items, store, settings)


def _run() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def _all_markdown(app: AppTest) -> str:
    return "\n".join(str(m.value) for m in app.markdown)


def test_app_starts_with_title_and_prompts_for_documents() -> None:
    app = _run()
    assert app.title[0].value == TITLE
    assert any("Upload and index at least one document" in str(i.value) for i in app.info)
    assert app.chat_input[0].disabled is True


def test_test_mode_banner_is_shown_for_offline_backends() -> None:
    app = _run()
    assert any("TEST MODE" in str(w.value) for w in app.warning)


def test_scenario_1_question_gets_answer_with_citation(offline_env: Settings, fixture_paths: dict[str, Path]) -> None:
    _index(offline_env, fixture_paths, "ml_pdf")
    app = _run()
    assert app.chat_input[0].disabled is False
    app.chat_input[0].set_value("What is machine learning according to this document?").run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert "subfield of artificial intelligence" in _all_markdown(app)
    labels = [e.label for e in app.expander]
    assert any(label.startswith("📚 Sources") for label in labels)
    assert any(label.startswith("🔎 Retrieved passages") for label in labels)
    assert "machine_learning.pdf" in _all_markdown(app) and "page 1" in _all_markdown(app)


def test_scenario_2_unanswerable_question_shows_refusal(offline_env: Settings, fixture_paths: dict[str, Path]) -> None:
    _index(offline_env, fixture_paths, "ml_pdf")
    app = _run()
    app.chat_input[0].set_value("Who won the football world cup in 2018?").run()
    assert REFUSAL_MESSAGE in _all_markdown(app)
    assert not any(e.label.startswith("📚 Sources") for e in app.expander)


def test_scenario_3_indexed_documents_survive_a_restart(offline_env: Settings, fixture_paths: dict[str, Path]) -> None:
    _index(offline_env, fixture_paths, "ml_pdf", "solar_txt")
    first = _run()
    assert "machine_learning.pdf" in _all_markdown(first.sidebar)

    st.cache_resource.clear()  # simulates restarting the server process
    restarted = _run()
    sidebar_text = _all_markdown(restarted.sidebar)
    assert "machine_learning.pdf" in sidebar_text and "solar_energy.txt" in sidebar_text
    restarted.chat_input[0].set_value("What is overfitting?").run()
    assert "page 3" in _all_markdown(restarted)


def test_scenario_4_answer_cites_both_documents(offline_env: Settings, fixture_paths: dict[str, Path]) -> None:
    _index(offline_env, fixture_paths, "ml_pdf", "solar_txt")
    app = _run()
    app.chat_input[0].set_value(
        "What is machine learning and how do solar panels convert sunlight into electricity?"
    ).run()
    text = _all_markdown(app)
    assert "machine_learning.pdf" in text and "solar_energy.txt" in text


def test_multiple_questions_in_one_session(offline_env: Settings, fixture_paths: dict[str, Path]) -> None:
    _index(offline_env, fixture_paths, "ml_pdf")
    app = _run()
    app.chat_input[0].set_value("What is supervised learning?").run()
    app.chat_input[0].set_value("What is overfitting?").run()
    assert len(app.chat_message) == 4


def test_clear_chat_button_empties_history(offline_env: Settings, fixture_paths: dict[str, Path]) -> None:
    _index(offline_env, fixture_paths, "ml_pdf")
    app = _run()
    app.chat_input[0].set_value("What is supervised learning?").run()
    assert len(app.chat_message) == 2
    next(b for b in app.sidebar.button if "Clear chat" in b.label).click().run()
    assert len(app.chat_message) == 0
    assert "machine_learning.pdf" in _all_markdown(app.sidebar), "clearing chat must not touch the index"


def test_clear_database_requires_confirmation_then_empties_index(
    offline_env: Settings, fixture_paths: dict[str, Path]
) -> None:
    _index(offline_env, fixture_paths, "ml_pdf")
    app = _run()
    clear_button = next(b for b in app.sidebar.button if "Clear DB" in b.label)
    assert clear_button.disabled is True

    app.sidebar.checkbox[0].check().run()
    next(b for b in app.sidebar.button if "Clear DB" in b.label).click().run()
    assert not app.exception
    assert "No documents indexed yet." in "\n".join(str(c.value) for c in app.sidebar.caption)
    assert app.chat_input[0].disabled is True

    store = VectorStoreManager(
        offline_env.chroma_dir, get_embeddings(offline_env), offline_env.collection_name, offline_env.embedding_id
    )
    assert store.count() == 0


def test_missing_api_key_shows_clear_instructions(monkeypatch: pytest.MonkeyPatch, fixture_paths: dict[str, Path]) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    st.cache_resource.clear()
    app = _run()
    errors = " ".join(str(e.value) for e in app.error)
    assert "GOOGLE_API_KEY" in errors and ".env" in errors
    assert app.chat_input[0].disabled is True


# ----------------------------------------------------------------------------
# Real upload flow: file_uploader widget -> "Process & index" button -> index
# ----------------------------------------------------------------------------
def _upload(app: AppTest, fixture_paths: dict[str, Path], *keys: str) -> None:
    files = [(fixture_paths[k].name, fixture_paths[k].read_bytes(), "application/octet-stream") for k in keys]
    app.sidebar.file_uploader[0].set_value(files).run()


def _click_index(app: AppTest) -> None:
    next(b for b in app.sidebar.button if "Process & index" in b.label).click().run()
    assert not app.exception, [e.value for e in app.exception]


def test_index_button_is_disabled_until_files_are_selected() -> None:
    app = _run()
    button = next(b for b in app.sidebar.button if "Process & index" in b.label)
    assert button.disabled is True


def test_upload_and_index_two_documents_then_ask_with_citations(fixture_paths: dict[str, Path]) -> None:
    app = _run()
    _upload(app, fixture_paths, "ml_pdf", "solar_txt")
    assert "machine_learning.pdf" in _all_markdown(app.sidebar)  # listed as "ready to index"
    _click_index(app)

    successes = " ".join(str(s.value) for s in app.sidebar.success)
    assert "machine_learning.pdf" in successes and "solar_energy.txt" in successes
    assert app.chat_input[0].disabled is False

    app.chat_input[0].set_value("What is machine learning according to this document?").run()
    text = _all_markdown(app)
    assert "subfield of artificial intelligence" in text and "machine_learning.pdf" in text


def test_upload_reports_corrupted_and_empty_files_but_indexes_the_rest(fixture_paths: dict[str, Path]) -> None:
    app = _run()
    _upload(app, fixture_paths, "corrupted_pdf", "empty_txt", "solar_txt")
    _click_index(app)
    errors = " ".join(str(e.value) for e in app.sidebar.error)
    assert "corrupted.pdf" in errors and "empty.txt" in errors
    assert "solar_energy.txt" in " ".join(str(s.value) for s in app.sidebar.success)
    assert app.chat_input[0].disabled is False


def test_uploading_the_same_file_again_is_reported_as_duplicate(fixture_paths: dict[str, Path]) -> None:
    app = _run()
    _upload(app, fixture_paths, "solar_txt")
    _click_index(app)
    _upload(app, fixture_paths, "solar_txt")
    _click_index(app)
    assert any("Already indexed" in str(i.value) for i in app.sidebar.info)


def test_additional_upload_keeps_previously_indexed_documents(fixture_paths: dict[str, Path]) -> None:
    app = _run()
    _upload(app, fixture_paths, "ml_pdf")
    _click_index(app)
    _upload(app, fixture_paths, "solar_txt")
    _click_index(app)
    sidebar_text = _all_markdown(app.sidebar)
    assert "machine_learning.pdf" in sidebar_text and "solar_energy.txt" in sidebar_text
