# Document Chatbot (RAG)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-20%20passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-84%25-brightgreen.svg)](tests/)

A production-ready RAG (Retrieval-Augmented Generation) system that lets you upload PDF documents and ask questions about their contents, with answers grounded in your documents and source citations.

**Demo Video:** [Watch Demo](https://1drv.ms/v/c/fe2fb007f7f25e16/IQCwsS6JsPUnR4ZLU4HYheIdASnRVbCcgm_nkaoDTUpC-YE?e=ethirp)

---

## Features

- 📄 **PDF Upload & Ingestion** — Upload any text-based PDF and ingest it into a persistent vector database
- 🔍 **Semantic Search** — Questions are embedded and matched against document chunks using cosine similarity
- 🤖 **Grounded Answers** — Groq LLM generates answers strictly from retrieved document context, never from general knowledge
- 📌 **Source Citations** — Every answer includes the source document it was retrieved from
- 💾 **Persistent Vector Store** — ChromaDB persists embeddings to disk; uploaded documents survive restarts
- 🐳 **Docker Ready** — Full multi-container setup with docker-compose
- 📊 **Structured Logging** — All pipeline steps logged with timing for observability
- ⚠️ **Graceful Error Handling** — Meaningful HTTP error codes for invalid files, empty queries, LLM timeouts, and rate limits

---

## Known Limitations

RAG retrieval performs best on **specific, localized factual questions** — questions answerable from a single document section.

**Questions that work well:**

- "What is the refund policy?"
- "When was the company founded?"
- "What are the system requirements?"

**Questions that work poorly:**

- "What is the main argument of this document?" — requires synthesizing the entire document; naive RAG retrieves only (altho can be increased) 4 chunks
- "Summarize everything" — same issue; top-k retrieval cannot cover full-document context
- "Compare section 2 and section 5" — multi-section synthesis not supported

This is a known limitation of naive RAG (top-k chunk retrieval). The fix is document summary indexing; storing a high-level summary alongside chunk embeddings for broad questions. This is tracked as a future enhancement.

Image-based (scanned) PDFs are not supported; text extraction requires text-based PDFs only.

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
Similarity Search → Top-4 Chunks + Similarity Scores + Source Metadata
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
| Testing          | pytest, pytest-cov (84% coverage)        |
| Logging          | Python logging with file output          |

---

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
git clone https://github.com/perlathebian/document-chatbot-rag.git
cd document-chatbot-rag
cp .env.example .env
# Add your GROQ_API_KEY to .env
docker-compose up
```

Access at `http://localhost:8501`

### Option 2: Manual (2 terminals)

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

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=term

# View HTML coverage report
pytest tests/ --cov=backend --cov-report=html
start htmlcov/index.html  # Windows
open htmlcov/index.html   # Mac
```

**Coverage: 84% (20 tests passing)**

---

## Logging & Error Handling

All pipeline activity is logged to `logs/rag_pipeline.log` with timestamps:

- PDF upload received (filename, file size)
- Text extraction result (character count)
- Chunks created (count per document)
- ChromaDB storage confirmation
- Query received (question text, source filter)
- Retrieval scores for top-4 chunks (similarity scores logged per chunk)
- LLM response time

**HTTP Error Codes:**

- `400` — Non-PDF file upload or image-based PDF
- `400` — Empty query
- `429` — Groq API rate limit exceeded
- `504` — LLM request timeout

---

## Project Structure

```
document-chatbot-rag/
├── backend/
│   ├── main.py           # FastAPI endpoints + error handling
│   ├── rag_pipeline.py   # Core RAG logic (ingest, retrieve, generate)
│   └── utils.py          # PDF extraction, chunking, ChromaDB client
├── frontend/
│   └── app.py            # Streamlit UI
├── tests/
│   └── test_pipeline.py  # 20 tests, 84% coverage
├── data/
│   ├── uploads/          # Uploaded PDFs (runtime)
│   └── sample_pdfs/      # Test documents
├── logs/                 # Application logs (runtime)
├── db/                   # ChromaDB vector store (persistent)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Author

**Perla Thebian**

- GitHub: [@perlathebian](https://github.com/perlathebian)
- LinkedIn: [Perla Thebian](https://www.linkedin.com/in/perla-thebian/)

Built as a portfolio project demonstrating production-ready RAG system design and full-stack ML engineering.
