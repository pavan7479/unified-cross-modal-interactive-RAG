# 📚 Unified Cross-Modal Interactive RAG System

A modular, high-performance Retrieval-Augmented Generation (RAG) system built with **Streamlit**, **LangChain**, and **Google Gemini 2.5 Flash**. This project enables interactive querying across various data modalities including local documents, YouTube videos, and live web scrapes.

## 🚀 Key Features
- **Multi-Source Ingestion**: Support for PDF, TXT, DOCX, CSV, YouTube transripts, and Web crawling.
- **Hybrid Retrieval Engine**: Combines **FAISS** (Semantic Search) and **BM25** (Keyword Search) with Reciprocal Rank Fusion.
- **Persistent Context**: Integrated short-term chat memory for follow-up questions.
- **Source Transparency**: Real-time source attribution for every answer generated.
- **High Efficiency**: Optimized for low-latency using `gemini-2.5-flash`.

## 🛠️ Technical Stack
- **AI Models**: Google Gemini 2.5 Flash (LLM), Gemini Embeddings.
- **Vector DB**: FAISS (Facebook AI Similarity Search).
- **Frontend**: Streamlit.
- **Orchestration**: LangChain.

## ⚙️ Setup & Installation
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd cross_modal_rag
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Environment Configuration**:
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```
4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```


