"""Document Intelligence - RAG Assistant (Streamlit app).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from config import ConfigurationError, Settings, get_settings
from src.embeddings import EmbeddingError, get_embeddings
from src.indexer import DUPLICATE, FAILED, INDEXED, UploadedItem, index_uploads
from src.qa_chain import GroundedQAChain, QAError, QAResult, build_llm
from src.retriever import ThresholdRetriever
from src.utils import clear_directory, markdown_safe
from src.vector_store import VectorStoreError, VectorStoreManager

APP_TITLE = "Document Intelligence – RAG Assistant"

st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1100px;}
      .app-subtitle {color: #6b7280; margin-top: -0.6rem; margin-bottom: 1.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the embedding model (the first run downloads it)…")
def load_store() -> tuple[VectorStoreManager | None, str | None]:
    """Open the persistent vector store. Returns ``(store, error_message)``."""
    settings = get_settings()
    try:
        embeddings = get_embeddings(settings)
        store = VectorStoreManager(
            persist_dir=settings.chroma_dir,
            embeddings=embeddings,
            collection_name=settings.collection_name,
            embedding_id=settings.embedding_id,
        )
        return store, None
    except (EmbeddingError, VectorStoreError) as exc:
        return None, str(exc)


@st.cache_resource(show_spinner=False)
def load_llm() -> tuple[object | None, str | None]:
    """Create the chat model. Returns ``(llm, error_message)``."""
    try:
        return build_llm(get_settings()), None
    except ConfigurationError as exc:
        return None, str(exc)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "index_results" not in st.session_state:
    st.session_state.index_results = []


def render_result(result: QAResult) -> None:
    """Show an assistant answer with expandable sources and retrieved snippets."""
    st.markdown(markdown_safe(result.answer))
    if not result.grounded:
        if result.best_score is not None and result.threshold is not None:
            st.caption(
                f"Best match similarity {result.best_score:.2f} · relevance threshold {result.threshold:.2f}"
            )
        return

    with st.expander(f"📚 Sources ({len(result.citations)})"):
        for citation in result.citations:
            score = f" · similarity {citation.score:.2f}" if citation.score is not None else ""
            st.markdown(f"**[{citation.index}] {markdown_safe(citation.source)}** — {citation.page_label}{score}")
            st.caption(markdown_safe(citation.snippet))
    with st.expander(f"🔎 Retrieved passages ({len(result.retrieved)})"):
        for citation in result.retrieved:
            cited = "cited" if any(c.index == citation.index for c in result.citations) else "not cited"
            st.markdown(f"**[{citation.index}] {markdown_safe(citation.source)}**, {citation.page_label} _({cited})_")
            st.caption(markdown_safe(citation.snippet))


def clear_database(store: VectorStoreManager, settings: Settings) -> None:
    """Remove the index and the stored copies of uploaded files."""
    store.clear()
    clear_directory(settings.upload_dir)
    st.session_state.index_results = []


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
settings = get_settings()
store, store_error = load_store()
llm, llm_error = load_llm()

st.title(APP_TITLE)
st.markdown(
    '<div class="app-subtitle">Upload PDF or TXT files, index them, then ask questions. '
    "Answers come only from your documents and always cite the file and page.</div>",
    unsafe_allow_html=True,
)

if store is None:
    st.error(f"The document index could not be opened.\n\n{store_error}")
    st.stop()

with st.sidebar:
    st.header("1 · Upload documents")
    uploaded_files = st.file_uploader(
        "PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        help=f"Up to {settings.max_upload_mb} MB per file.",
    )
    if uploaded_files:
        st.caption("Ready to index:")
        for uploaded in uploaded_files:
            st.markdown(f"- {uploaded.name}")

    index_clicked = st.button(
        "⚙️ Process & index documents",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if index_clicked and uploaded_files:
        items = [UploadedItem(name=f.name, data=f.getvalue()) for f in uploaded_files]
        progress_bar = st.progress(0.0, text="Starting…")
        with st.status("Indexing documents…", expanded=True) as status:

            def on_progress(fraction: float, message: str) -> None:
                progress_bar.progress(min(fraction, 1.0), text=message)
                status.write(message)

            results = index_uploads(items, store, settings, progress=on_progress)
            failures = sum(r.status == FAILED for r in results)
            status.update(
                label="Indexing finished" if not failures else f"Indexing finished with {failures} problem(s)",
                state="complete" if not failures else "error",
                expanded=False,
            )
        progress_bar.empty()
        st.session_state.index_results = results
        st.session_state.uploader_key += 1
        st.rerun()

    for result in st.session_state.index_results:
        text = f"**{result.filename}** — {result.message}"
        if result.status == INDEXED:
            st.success(text, icon="✅")
        elif result.status == DUPLICATE:
            st.info(text, icon="♻️")
        else:
            st.error(text, icon="⚠️")

    st.divider()
    st.header("2 · Indexed documents")
    documents = store.list_documents()
    if documents:
        for doc in documents:
            unit = "page" if doc.file_type == "pdf" else "section"
            st.markdown(f"📄 **{doc.source}**  \n{doc.chunk_count} chunks · {doc.page_count} {unit}(s)")
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    with st.expander("Retrieval settings"):
        top_k = st.slider("Passages to retrieve (top-k)", 1, 10, settings.top_k)
        threshold = st.slider(
            "Relevance threshold (cosine similarity)",
            -1.0,
            1.0,
            float(settings.relevance_threshold),
            0.01,
            help="Passages scoring below this are ignored. Scores depend on the embedding model, "
            "so re-tune this if you change it.",
        )

    st.divider()
    col_chat, col_db = st.columns(2)
    if col_chat.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    confirm = st.checkbox("Confirm database deletion")
    if col_db.button("🗑️ Clear DB", use_container_width=True, disabled=not confirm):
        try:
            clear_database(store, settings)
            st.session_state.messages = []
            st.rerun()
        except VectorStoreError as exc:
            st.error(str(exc))

    st.caption(f"Embeddings: `{settings.embedding_backend}` · LLM: `{settings.llm_provider}`")

# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------
if settings.llm_provider == "mock":
    st.warning(
        "Offline TEST MODE: answers are extracted sentences from your documents, not generated by an AI model. "
        "Set LLM_PROVIDER=gemini and add GOOGLE_API_KEY in .env for real answers.",
        icon="🧪",
    )
if settings.embedding_backend == "hash":
    st.warning("Offline TEST MODE: keyword-hash embeddings are in use instead of a sentence-transformers model.", icon="🧪")

if llm is None:
    st.error(llm_error or "The language model is not configured.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(markdown_safe(message["content"]))
        else:
            render_result(message["result"])

if not documents:
    st.info("👈 Upload and index at least one document to start asking questions.")

question = st.chat_input(
    "Ask a question about your documents…",
    disabled=llm is None or not documents,
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(markdown_safe(question))
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents…"):
            try:
                retriever = ThresholdRetriever(store=store, k=top_k, score_threshold=threshold)
                chain = GroundedQAChain(retriever, llm, snippet_chars=settings.snippet_chars)
                qa_result = chain.ask(question)
            except (QAError, EmbeddingError, VectorStoreError) as exc:
                st.error(f"Something went wrong while answering: {exc}")
            else:
                render_result(qa_result)
                st.session_state.messages.append({"role": "assistant", "result": qa_result})
