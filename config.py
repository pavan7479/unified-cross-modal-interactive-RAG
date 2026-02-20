import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ---------------------------
    # API
    # ---------------------------
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    # ---------------------------
    # Models
    # ---------------------------
    LLM_MODEL = "models/gemini-2.5-flash"
    EMBEDDING_MODEL = "gemini-embedding-001"

    LLM_TEMPERATURE = 0.2
    LLM_MAX_TOKENS = 2048  # Lower for latency

    # ---------------------------
    # Chunking (optimized for latency)
    # ---------------------------
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 150


    # ---------------------------
    # Retrieval
    # ---------------------------
    RETRIEVAL_K = 5         # Final top documents sent to LLM
    FETCH_K = 15             # For MMR diversity
    SIMILARITY_THRESHOLD = 0.45  # More realistic threshold

    # Hybrid Weights
    DENSE_WEIGHT = 0.75
    BM25_WEIGHT = 0.25

    # ---------------------------
    # Interactive Memory
    # ---------------------------
    MAX_CHAT_HISTORY = 5  # Keep last N messages only

    # ---------------------------
    # Web Crawling (Optional)
    # ---------------------------
    MAX_CRAWL_DEPTH = 0      # was maybe 1 or 2
    MAX_CRAWL_PAGES = 5      # keep small
    REQUEST_TIMEOUT = 15
