import streamlit as st

from ingestion.loaders import load_uploaded_file, load_youtube, load_web
from ingestion.chunking import chunk_documents
from ingestion.vectorstore import build_indices, add_documents
from retrieval.pipeline import run_rag_pipeline
from utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Unified Interactive RAG", layout="wide")

st.title("📚 Unified Interactive Cross-Modal RAG System")

# ==========================================================
# Session State Initialization
# ==========================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "bm25" not in st.session_state:
    st.session_state.bm25 = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==========================================================
# Sidebar - Data Ingestion
# ==========================================================

st.sidebar.header("Upload or Add Sources")

uploaded_files = st.sidebar.file_uploader(
    "Upload Files (PDF, TXT, DOCX, CSV)",
    type=["pdf", "txt", "docx", "csv"],
    accept_multiple_files=True,
)

youtube_url = st.sidebar.text_input("YouTube URL")
web_url = st.sidebar.text_input("Website URL")

if st.sidebar.button("Process Sources"):

    all_docs = []

    try:
        # -------------------------
        # File Uploads
        # -------------------------
        if uploaded_files:
            for file in uploaded_files:
                docs = load_uploaded_file(file)
                all_docs.extend(docs)

        # -------------------------
        # YouTube
        # -------------------------
        if youtube_url:
            yt_docs = load_youtube(youtube_url)
            all_docs.extend(yt_docs)

        # -------------------------
        # Web
        # -------------------------
        if web_url:
            web_docs = load_web(web_url)
            all_docs.extend(web_docs)

        if not all_docs:
            st.sidebar.warning("No sources provided.")
        else:
            # Chunking
            chunked_docs = chunk_documents(all_docs)

            if st.session_state.vectorstore is None:
                vectorstore, bm25 = build_indices(chunked_docs)
                st.session_state.vectorstore = vectorstore
                st.session_state.bm25 = bm25
            else:
                vectorstore, bm25 = add_documents(
                    st.session_state.vectorstore,
                    st.session_state.bm25,
                    chunked_docs,
                )
                st.session_state.vectorstore = vectorstore
                st.session_state.bm25 = bm25

            st.sidebar.success("Sources processed successfully!")

    except Exception as e:
        st.sidebar.error(f"Error processing sources: {e}")


# ==========================================================
# Chat Interface
# ==========================================================

st.subheader("💬 Ask Questions")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Ask something about your sources...")

if user_input:

    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            if st.session_state.vectorstore is None:
                answer = "No documents loaded."
                sources = []
            else:
                result = run_rag_pipeline(
                    query=user_input,
                    vectorstore=st.session_state.vectorstore,
                    bm25=st.session_state.bm25,
                    chat_history=st.session_state.chat_history,
                )

                answer = result["answer"]
                sources = result["sources"]

            st.markdown(answer)

            if sources:
                st.markdown("**Sources:**")
                for src in sources:
                    st.markdown(
                        f"- {src.get('type')} | {src.get('source')} | Page: {src.get('page')}"
                    )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )


# ==========================================================
# Reset Session
# ==========================================================

if st.sidebar.button("Reset Session"):
    st.session_state.clear()
    st.rerun()
