from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from config import Config
from ingestion.embeddings import get_embedding_model
from utils.logger import get_logger

logger = get_logger(__name__)


# ==========================================================
# Build In-Memory Indices (Session-Based)
# ==========================================================

def build_indices(
    documents: List[Document],
) -> Tuple[FAISS, BM25Retriever]:
    """
    Builds in-memory:
    - FAISS vector index (semantic search)
    - BM25 retriever (keyword search)

    No disk persistence.
    Designed for interactive RAG.
    """

    if not documents:
        raise ValueError("No documents provided for indexing.")

    logger.info(f"Building indices for {len(documents)} chunks")

    embeddings = get_embedding_model()

    # ---------------------------
    # Dense Vector Index (FAISS)
    # ---------------------------
    vectorstore = FAISS.from_documents(
        documents,
        embedding=embeddings,
    )

    # ---------------------------
    # Keyword Index (BM25)
    # ---------------------------
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = Config.FETCH_K  # Fetch more before hybrid fusion

    logger.info("FAISS + BM25 indices built successfully")

    return vectorstore, bm25


# ==========================================================
# Add Documents to Existing Indices
# ==========================================================

def add_documents(
    vectorstore: FAISS,
    bm25: BM25Retriever,
    new_documents: List[Document],
) -> Tuple[FAISS, BM25Retriever]:
    """
    Adds new documents to in-memory indices.

    FAISS supports incremental updates.
    BM25 must be rebuilt.
    """

    if not new_documents:
        return vectorstore, bm25

    logger.info(f"Adding {len(new_documents)} new chunks")

    # Update FAISS
    vectorstore.add_documents(new_documents)

    # Rebuild BM25 (required — no incremental update support)
    all_docs = bm25.docs + new_documents
    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = Config.FETCH_K

    logger.info("Indices updated successfully")

    return vectorstore, bm25


# ==========================================================
# Dense Retriever (MMR for Diversity)
# ==========================================================

def get_dense_retriever(vectorstore: FAISS):
    """
    Returns MMR-based dense retriever.
    Helps reduce redundancy.
    """

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": Config.RETRIEVAL_K,
            "fetch_k": Config.FETCH_K,
        },
    )
