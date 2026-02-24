# Document Chatbot (RAG)

A production-ready RAG (Retrieval-Augmented Generation) system that lets you upload PDF documents and ask questions about their contents, with answers grounded in your documents and source citations.

**[Live Demo →](https://huggingface.co/spaces/perlathebian/document-chatbot-rag)**

---

## Architecture

```
INGESTION PIPELINE
──────────────────
PDF Upload
    │
    ▼
Text Extraction (pdfplumber)
    │
    ▼
Text Chunking (RecursiveCharacterTextSplitter, 500 chars / 50 overlap)
    │
    ▼
Embeddings (sentence-transformers: all-MiniLM-L6-v2)
    │
    ▼
Vector Storage (ChromaDB persistent)

RETRIEVAL PIPELINE
──────────────────
User Question
    │
    ▼
Question Embedding
    │
    ▼
Similarity Search → Top-4 Chunks + Source Metadata
    │
    ▼
Prompt Construction (Context + Question + System Instructions)
    │
    ▼
Groq LLM (llama-3.3-70b-versatile)
    │
    ▼
Answer + Source Citations
```

---

## Tech Stack

| Component        | Tool                                     |
| ---------------- | ---------------------------------------- |
| PDF Extraction   | pdfplumber                               |
| Text Splitting   | LangChain RecursiveCharacterTextSplitter |
| Embeddings       | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database  | ChromaDB (persistent)                    |
| LLM              | Groq API (llama-3.3-70b-versatile)       |
| Orchestration    | LangChain                                |
| Backend API      | FastAPI                                  |
| Frontend         | Streamlit                                |
| Containerization | Docker                                   |

---

## Run Locally

**1. Clone and set up environment:**

```bash
git clone https://github.com/perlathebian/document-chatbot-rag.git
cd document-chatbot-rag
python3.11 -m venv venv
venv\Scripts\activate     # Mac: source venv/bin/activate
pip install -r requirements.txt
```

**2. Configure environment:**

```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

**3. Start the backend:**

```bash
uvicorn backend.main:app --reload --port 8000
```

**4. Start the frontend (new terminal):**

```bash
streamlit run frontend/app.py
```

**5. Open** `http://localhost:8501`

---

## API Endpoints

| Method | Endpoint  | Description                          |
| ------ | --------- | ------------------------------------ |
| GET    | `/health` | Health check                         |
| POST   | `/upload` | Upload and ingest a PDF              |
| POST   | `/query`  | Ask a question, get answer + sources |

API docs available at `/docs` (Swagger UI).

---

## Known Limitations

RAG retrieval performs best on specific factual questions localized within a document section.General questions requiring full-document synthesis (e.g. "what is the main argument?") may not retrieve sufficient context. This is a known limitation of naive RAG, addressable with document summary indexing in future iterations.

Image-based (scanned) PDFs are not supported; text extraction requires text-based PDFs.

---

## Project Structure

```
document-chatbot-rag/
├── backend/
│   ├── main.py           # FastAPI endpoints
│   ├── rag_pipeline.py   # Core RAG logic
│   └── utils.py          # PDF extraction, chunking
├── frontend/
│   └── app.py            # Streamlit UI
├── tests/
│   └── test_pipeline.py  # Pytest unit tests
├── data/
│   └── sample_pdfs/      # Test documents
├── Dockerfile
└── requirements.txt
```
