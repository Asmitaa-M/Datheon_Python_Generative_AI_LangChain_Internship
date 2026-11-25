"""Streamlit RAG chatbot with streaming output, chat history and session state.

Run:  streamlit run app.py
Needs: ANTHROPIC_API_KEY (env var or entered in the sidebar).
"""
import os
import tempfile

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="RAG Chat", page_icon="💬", layout="wide")

# --------------------------------------------------------------------------
# Session state: initialise once per browser session
# --------------------------------------------------------------------------
def init_state():
    st.session_state.setdefault("messages", [])        # [{"role","content","sources"}]
    st.session_state.setdefault("vectorstore", None)   # InMemoryVectorStore | None
    st.session_state.setdefault("indexed_files", [])   # names already embedded


init_state()


def reset_chat():
    st.session_state.messages = []


def reset_all():
    st.session_state.messages = []
    st.session_state.vectorstore = None
    st.session_state.indexed_files = []


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_llm(model: str, temperature: float, api_key: str) -> ChatAnthropic:
    return ChatAnthropic(
        model=model, temperature=temperature, api_key=api_key, streaming=True
    )


def index_files(files):
    """Load, split and embed uploaded files into the session's vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = []
    for f in files:
        suffix = os.path.splitext(f.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.getvalue())
            path = tmp.name
        try:
            loader = PyPDFLoader(path) if suffix == ".pdf" else TextLoader(path, encoding="utf-8")
            for d in loader.load():
                d.metadata["source"] = f.name
                docs.append(d)
        finally:
            os.unlink(path)
    chunks = splitter.split_documents(docs)
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = InMemoryVectorStore(get_embeddings())
    st.session_state.vectorstore.add_documents(chunks)
    st.session_state.indexed_files += [f.name for f in files]
    return len(chunks)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Anthropic API key", type="password", value=os.getenv("ANTHROPIC_API_KEY", "")
    )
    model = st.text_input("Model", value="claude-sonnet-5-5")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    top_k = st.slider("Chunks retrieved (k)", 1, 10, 4)
    max_turns = st.slider("History turns sent to model", 1, 20, 6)

    st.divider()
    st.header("Knowledge base")
    uploads = st.file_uploader(
        "Upload PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if st.button("Index documents", disabled=not uploads, use_container_width=True):
        new = [f for f in uploads if f.name not in st.session_state.indexed_files]
        if not new:
            st.info("Those files are already indexed.")
        else:
            with st.spinner("Embedding…"):
                n = index_files(new)
            st.success(f"Indexed {n} chunks from {len(new)} file(s).")

    if st.session_state.indexed_files:
        st.caption("Indexed: " + ", ".join(st.session_state.indexed_files))

    st.divider()
    c1, c2 = st.columns(2)
    c1.button("Clear chat", on_click=reset_chat, use_container_width=True)
    c2.button("Reset all", on_click=reset_all, use_container_width=True)

# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. If context is provided, ground your answer in it "
            "and say so when the context does not contain the answer. If no context is "
            "provided, answer from general knowledge.\n\nContext:\n{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Rewrite the user's last message as a standalone search query, "
                   "using the chat history for context. Output only the query."),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


def to_lc_history(messages, max_turns):
    msgs = messages[-(max_turns * 2):]
    return [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in msgs
    ]


def answer_stream(question: str, history, llm, sources_out: list):
    """Generator of answer tokens; fills sources_out as a side effect."""
    context = ""
    vs = st.session_state.vectorstore
    if vs is not None:
        query = question
        if history:  # make follow-up questions retrievable
            query = (CONDENSE_PROMPT | llm | StrOutputParser()).invoke(
                {"history": history, "question": question}
            )
        docs = vs.similarity_search(query, k=top_k)
        context = "\n\n".join(
            f"[{i+1}] ({d.metadata.get('source')}) {d.page_content}" for i, d in enumerate(docs)
        )
        sources_out.extend(
            {"source": d.metadata.get("source"), "page": d.metadata.get("page"),
             "text": d.page_content} for d in docs
        )

    chain = ANSWER_PROMPT | llm | StrOutputParser()
    yield from chain.stream(
        {"context": context or "(none)", "history": history, "question": question}
    )


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            page = f", p.{s['page'] + 1}" if isinstance(s.get("page"), int) else ""
            st.markdown(f"**[{i}] {s['source']}{page}**")
            st.caption(s["text"][:400] + ("…" if len(s["text"]) > 400 else ""))


# --------------------------------------------------------------------------
# Main chat UI
# --------------------------------------------------------------------------
st.title("💬 RAG Chat")
st.caption("Upload documents in the sidebar to ground answers, or just chat.")

# Replay history on every rerun
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_sources(m.get("sources"))

if prompt := st.chat_input("Ask something…"):
    if not api_key:
        st.error("Enter your Anthropic API key in the sidebar.")
        st.stop()

    history = to_lc_history(st.session_state.messages, max_turns)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        sources: list = []
        try:
            llm = get_llm(model, temperature, api_key)
            # st.write_stream renders tokens live and returns the full text
            reply = st.write_stream(answer_stream(prompt, history, llm, sources))
        except Exception as e:  # noqa: BLE001
            reply = f"⚠️ Error: {e}"
            st.error(reply)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )
