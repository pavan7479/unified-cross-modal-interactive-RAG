from typing import List, Dict

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import Config
from utils.logger import get_logger
from retrieval.hybrid import hybrid_retrieve, select_top_documents

logger = get_logger(__name__)

_llm_instance = None


# ==========================================================
# LLM Singleton
# ==========================================================

def get_llm():
    global _llm_instance

    if _llm_instance is None:
        _llm_instance = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            google_api_key=Config.GOOGLE_API_KEY,
            max_output_tokens=Config.LLM_MAX_TOKENS,
        )

    return _llm_instance


# ==========================================================
# Context Builder
# ==========================================================

def build_context(docs: List[Document]) -> str:
    return "\n\n".join([doc.page_content for doc in docs])


# ==========================================================
# Main RAG Pipeline (Interactive)
# ==========================================================

def run_rag_pipeline(
    query: str,
    vectorstore,
    bm25,
    chat_history: List[Dict[str, str]],
):
    """
    Interactive unified RAG pipeline.
    """

    logger.info("Running RAG pipeline")

    # ---------------------------------------
    # Step 1: Hybrid Retrieval
    # ---------------------------------------
    results = hybrid_retrieve(query, vectorstore, bm25)
    selected_docs = select_top_documents(results)

    if not selected_docs:
        return {
            "answer": "No relevant context found.",
            "sources": [],
        }

    # ---------------------------------------
    # Step 2: Build Context
    # ---------------------------------------
    context_text = build_context(selected_docs)

    # ---------------------------------------
    # Step 3: Add Short-Term Memory
    # ---------------------------------------
    memory_block = ""

    if chat_history:
        recent_history = chat_history[-Config.MAX_CHAT_HISTORY :]
        memory_lines = []

        for turn in recent_history:
            role = turn["role"]
            content = turn["content"]
            memory_lines.append(f"{role.upper()}: {content}")

        memory_block = "\n".join(memory_lines)

    # ---------------------------------------
    # Step 4: LLM Grounded Generation
    # ---------------------------------------
    llm = get_llm()

    system_prompt = (
    "You are a helpful and knowledgeable assistant.\n",
    "Use the provided context to answer the user's question.\n",
    "Base your answer primarily on the context.\n",
    "You may expand or clarify logically for better understanding.\n",
    "Do not introduce unrelated or unsupported information.\n",
    "If the context is insufficient, respond exactly with:\n",
    "'No relevant context found.'\n",
    "If the question requests detailed explanation (e.g., 5 or 10 marks), "
    "provide a well-structured and sufficiently detailed answer.\n",
    "Use conversation history to resolve follow-up references like 'it' or 'that'."
)


    user_prompt = (
        f"Conversation History:\n{memory_block}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question:\n{query}"
    )

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    answer = response.content.strip()

    # ---------------------------------------
    # Step 5: Source Formatting
    # ---------------------------------------
    sources = [
        {
            "source": doc.metadata.get("source"),
            "type": doc.metadata.get("source_type"),
            "page": doc.metadata.get("page"),
        }
        for doc in selected_docs
    ]

    return {
        "answer": answer,
        "sources": sources,
    }
