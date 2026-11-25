# Streamlit RAG Chat

A local web app: upload PDFs/text, then chat with streaming answers, history and cited sources.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # or paste it in the sidebar
streamlit run app.py
```
Opens at http://localhost:8501.

## How it works
- **Session state** (`st.session_state`): `messages` (history), `vectorstore` (in-memory index), `indexed_files`. History is replayed on each rerun; "Clear chat" and "Reset all" buttons manage it.
- **Streaming**: the LangChain chain (`prompt | ChatAnthropic | StrOutputParser`) is consumed via `chain.stream`, rendered live with `st.write_stream`.
- **RAG**: uploads are split (1000/150), embedded locally (`all-MiniLM-L6-v2`), and stored in an `InMemoryVectorStore`. Follow-up questions are rewritten into standalone queries using history before retrieval. Retrieved chunks are shown under each answer.
- With no documents indexed, it works as a plain chatbot.

## Customising
- Swap the LLM in `get_llm` (any LangChain chat model supports `.stream`).
- Swap `InMemoryVectorStore` for Chroma/FAISS for persistence.
- To wrap an agent instead, replace `answer_stream` with a generator that yields text from `agent.stream(...)`.
