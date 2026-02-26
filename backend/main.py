import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from backend.rag_pipeline import ingest_document, generate_answer

load_dotenv()

import logging
import time

logger = logging.getLogger(__name__)

# App setup
app = FastAPI(
    title="Document Chatbot API",
    description="RAG-based document Q&A — upload PDFs, ask questions, get sourced answers.",
    version="1.0.0"
)

# CORS (allows the Streamlit frontend to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_DIR = "data/uploads"

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    source: str = None      # optional: if provided, filters search to this document

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

class UploadResponse(BaseModel):
    status: str
    document: str
    chunks_stored: int

# Endpoints
@app.get("/health")
def health_check():
    """Basic health check to confirm API is running."""
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and ingest it into the vector database.
    Rejects non-PDF files with a 400 error.
    """
    logger.info(f"Upload received: {file.filename} (content_type={file.content_type})")

    # Validate file type
    if not file.filename.endswith(".pdf"):
        logger.warning(f"Rejected non-PDF upload: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Only PDF files are accepted. Got: {file.filename}"
        )

    # Save uploaded file to disk
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_size = os.path.getsize(file_path)
        logger.info(f"File saved: {file.filename} ({file_size / 1024:.1f} KB)")    
    except Exception as e:
        logger.error(f"Failed to save uploaded file {file.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {str(e)}"
        )

    # Run ingestion pipeline
    try:
        result = ingest_document(file_path, file.filename)
        logger.info(f"Ingestion complete: {file.filename} — {result['chunks_stored']} chunks")
    except ValueError as e:
        # ValueError means image-based PDF or unreadable file
        logger.warning(f"Image-based or unreadable PDF: {file.filename} — {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )

    return UploadResponse(
        status="success",
        document=result["document"],
        chunks_stored=result["chunks_stored"]
    )


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Ask a question about ingested documents.
    Optionally filter by source document name
    """
    logger.info(f"Query received: '{request.question[:80]}' (source_filter={request.source})")

    # Validate question is not empty
    if not request.question or not request.question.strip():
        logger.warning("Empty query rejected")
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    query_start = time.time()
    
    try:
        result = generate_answer(
            question=request.question.strip(),
            source_filter=request.source
        )
    
    except TimeoutError as e:
        logger.error(f"LLM timeout for query: '{request.question[:80]}'")
        raise HTTPException(
            status_code=504,
            detail="LLM request timed out. Please try again."
        )    
    
    except RuntimeError as e:
        if "rate limit" in str(e).lower():
            logger.error(f"LLM rate limit hit for query: '{request.question[:80]}'")
            raise HTTPException(
                status_code=429,
                detail="LLM rate limit exceeded. Please wait a moment and try again."
            )
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed: {str(e)}"
        )
    
    elapsed = time.time() - query_start
    logger.info(f"Query answered in {elapsed:.2f}s — sources: {result['sources']}")

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"]
    )