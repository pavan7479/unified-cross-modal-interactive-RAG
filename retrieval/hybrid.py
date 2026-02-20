from typing import List, Tuple
from collections import defaultdict

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


# ==========================================================
# Hybrid Retrieval (Dense + BM25)
# ==========================================================

def hybrid_retrieve(
    query: str,
    vectorstore: FAISS,
    bm25: BM25Retriever,
) -> List[Tuple[Document, float]]:
    """
    Performs hybrid retrieval using:
    - Dense FAISS (MMR)
    - BM25 keyword retrieval

    Returns:
        List of (Document, combined_score)
    """

    logger.info("Starting hybrid retrieval")

    # ---------------------------
    # Dense Retrieval (MMR)
    # ---------------------------
    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": Config.FETCH_K,
            "fetch_k": Config.FETCH_K,
        },
    )

    dense_docs = dense_retriever.invoke(query)

    # ---------------------------
    # BM25 Retrieval
    # ---------------------------
    bm25_docs = bm25.invoke(query)

    # ---------------------------
    # Score Fusion (Rank-Based)
    # ---------------------------
    scores = defaultdict(float)
    doc_lookup = {}

    # Dense scoring
    for rank, doc in enumerate(dense_docs):
        key = (doc.page_content, tuple(doc.metadata.items()))
        doc_lookup[key] = doc
        scores[key] += Config.DENSE_WEIGHT * (1 / (rank + 1))

    # BM25 scoring
    for rank, doc in enumerate(bm25_docs):
        key = (doc.page_content, tuple(doc.metadata.items()))
        doc_lookup[key] = doc
        scores[key] += Config.BM25_WEIGHT * (1 / (rank + 1))

    # Combine and sort
    combined = [
        (doc_lookup[key], score)
        for key, score in scores.items()
    ]

    combined.sort(key=lambda x: x[1], reverse=True)

    logger.info(f"Hybrid retrieval returned {len(combined)} results")

    return combined


# ==========================================================
# Apply Threshold + Limit
# ==========================================================

def select_top_documents(
    results: List[Tuple[Document, float]]
) -> List[Document]:
    """
    Applies threshold filtering and returns top documents.
    """

    if not results:
        return []

    top_score = results[0][1]

    if top_score < Config.SIMILARITY_THRESHOLD:
        logger.info("Top score below threshold. No relevant context found.")
        return []

    # Return only top K
    return [
        doc for doc, score in results[: Config.RETRIEVAL_K]
    ]
