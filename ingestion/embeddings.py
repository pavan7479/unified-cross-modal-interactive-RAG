from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

_embedding_instance = None


def get_embedding_model():
    """
    Returns a singleton instance of GoogleGenerativeAIEmbeddings.
    Ensures only one embedding client is created.
    """
    global _embedding_instance

    if _embedding_instance is None:
        try:
            logger.info(
                f"Initializing Gemini Embeddings: {Config.EMBEDDING_MODEL}"
            )

            _embedding_instance = GoogleGenerativeAIEmbeddings(
                model=Config.EMBEDDING_MODEL,
                google_api_key=Config.GOOGLE_API_KEY,
            )

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise e

    return _embedding_instance
