import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from backend.rag_pipeline import ingest_document, generate_answer

load_dotenv()

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
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Only PDF files are accepted. Got: {file.filename}"
        )

    # Save uploaded file to disk
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {str(e)}"
        )

    # Run ingestion pipeline
    try:
        result = ingest_document(file_path, file.filename)
    except ValueError as e:
        # ValueError means image-based PDF or unreadable file
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
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
    # Validate question is not empty
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        result = generate_answer(
            question=request.question.strip(),
            source_filter=request.source
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed: {str(e)}"
        )

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"]
    )