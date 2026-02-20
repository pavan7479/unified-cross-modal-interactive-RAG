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
## sample output
**Empty UI**
<img width="1366" height="768" alt="Screenshot 2026-02-21 012501" src="https://github.com/user-attachments/assets/2715b25b-48d3-4378-8854-c483113eb069" />

**Sample input & Output**
<img width="1366" height="768" alt="Screenshot 2026-02-21 012739" src="https://github.com/user-attachments/assets/30a6b7f7-eb9f-4760-8226-1d1fd97b6e2a" />

**Source indication**
<img width="1366" height="768" alt="Screenshot 2026-02-21 012833" src="https://github.com/user-attachments/assets/c0a11236-332b-4ed6-9326-70cffe7d4411" />

