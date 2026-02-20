import uuid
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


# ==========================================================
# Recursive Chunking (Optimized for Hybrid Retrieval)
# ==========================================================

def get_splitter() -> RecursiveCharacterTextSplitter:
    """
    Configured recursive text splitter.
    Optimized for semantic + keyword hybrid retrieval.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
        strip_whitespace=True,
    )


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits documents into chunks and preserves metadata.
    """

    logger.info(f"Chunking {len(documents)} documents")

    splitter = get_splitter()
    chunks = splitter.split_documents(documents)
    # SAFETY LIMIT
    MAX_TOTAL_CHUNKS = 100
    if len(chunks) > MAX_TOTAL_CHUNKS:
        logger.warning(f"Too many chunks ({len(chunks)}). Limiting to {MAX_TOTAL_CHUNKS}.")
        chunks = chunks[:MAX_TOTAL_CHUNKS]


    enriched_chunks = []

    for chunk in chunks:
        metadata = chunk.metadata.copy()

        # Ensure consistent metadata
        metadata["chunk_id"] = str(uuid.uuid4())

        enriched_chunks.append(
            Document(
                page_content=chunk.page_content.strip(),
                metadata=metadata,
            )
        )

    logger.info(f"Created {len(enriched_chunks)} chunks")

    return enriched_chunks
