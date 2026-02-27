import pytest

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import io
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils import extract_text, chunk_text

# extract_text tests
def test_extract_text_returns_string():
    """extract_text should return a string for a valid PDF."""
    pdf_path = "data/sample_pdfs/Stop Saying AI.pdf"
    result = extract_text(pdf_path)
    assert isinstance(result, str)

def test_extract_text_not_empty():
    """extract_text should return non-empty content for a text-based PDF."""
    pdf_path = "data/sample_pdfs/Stop Saying AI.pdf"
    result = extract_text(pdf_path)
    assert len(result) > 100

# chunk_text tests
def test_chunk_text_returns_list():
    """chunk_text should return a list of strings."""
    sample_text = "This is a test sentence. " * 100
    chunks = chunk_text(sample_text)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)

def test_chunk_text_respects_size():
    """No chunk should exceed chunk_size + overlap (500 + 50 = 550 characters)."""
    sample_text = "This is a test sentence. " * 200
    chunks = chunk_text(sample_text)
    for chunk in chunks:
        assert len(chunk) <= 550, f"Chunk too large: {len(chunk)} characters"

def test_chunk_text_not_empty_chunks():
    """chunk_text should not produce empty chunks."""
    sample_text = "This is a test sentence. " * 100
    chunks = chunk_text(sample_text)
    assert all(len(chunk) > 0 for chunk in chunks)


# Utils.py: fix missing get_chroma_client coverage
def test_get_chroma_client_returns_client():
    """get_chroma_client should return a ChromaDB client instance."""
    from backend.utils import get_chroma_client
    client = get_chroma_client()
    assert client is not None



# main.py: API endpoint tests using FastAPI TestClient
@pytest.fixture
def client():
    """FastAPI test client."""
    from backend.main import app
    return TestClient(app)


def test_health_check(client):
    """GET /health should return 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_rejects_non_pdf(client):
    """POST /upload should return 400 for non-PDF files."""
    fake_file = io.BytesIO(b"not a pdf")
    response = client.post(
        "/upload",
        files={"file": ("document.txt", fake_file, "text/plain")}
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_upload_valid_pdf(client):
    """POST /upload should return 200 for valid PDF with mocked ingestion."""
    with patch("backend.main.ingest_document") as mock_ingest:
        mock_ingest.return_value = {
            "document": "test.pdf",
            "chunks_stored": 5,
            "status": "success"
        }
        pdf_path = "data/sample_pdfs/Stop Saying AI.pdf"
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("test.pdf", f, "application/pdf")}
            )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_stored"] == 5


def test_upload_image_pdf_returns_400(client):
    """POST /upload should return 400 for image-based PDF."""
    with patch("backend.main.ingest_document") as mock_ingest:
        mock_ingest.side_effect = ValueError("Could not extract text. PDF may be image-based.")
        pdf_path = "data/sample_pdfs/Stop Saying AI.pdf"
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("test.pdf", f, "application/pdf")}
            )
    assert response.status_code == 400


def test_query_empty_question_returns_400(client):
    """POST /query should return 400 for empty question."""
    response = client.post(
        "/query",
        json={"question": "   "}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_query_valid_question(client):
    """POST /query should return 200 with answer and sources."""
    with patch("backend.main.generate_answer") as mock_answer:
        mock_answer.return_value = {
            "answer": "The document discusses AI trends.",
            "sources": ["test.pdf"]
        }
        response = client.post(
            "/query",
            json={"question": "What is this document about?"}
        )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert data["answer"] == "The document discusses AI trends."


def test_query_timeout_returns_504(client):
    """POST /query should return 504 on LLM timeout."""
    with patch("backend.main.generate_answer") as mock_answer:
        mock_answer.side_effect = TimeoutError("LLM timed out")
        response = client.post(
            "/query",
            json={"question": "What is this about?"}
        )
    assert response.status_code == 504


def test_query_rate_limit_returns_429(client):
    """POST /query should return 429 on rate limit error."""
    with patch("backend.main.generate_answer") as mock_answer:
        mock_answer.side_effect = RuntimeError("rate limit exceeded")
        response = client.post(
            "/query",
            json={"question": "What is this about?"}
        )
    assert response.status_code == 429


def test_query_with_source_filter(client):
    """POST /query should pass source filter to generate_answer."""
    with patch("backend.main.generate_answer") as mock_answer:
        mock_answer.return_value = {
            "answer": "Filtered answer.",
            "sources": ["specific.pdf"]
        }
        response = client.post(
            "/query",
            json={"question": "What is this about?", "source": "specific.pdf"}
        )
    assert response.status_code == 200
    mock_answer.assert_called_once_with(
        question="What is this about?",
        source_filter="specific.pdf"
    )



# rag_pipelin.py: mock-based tests
def test_ingest_document_success():
    """ingest_document should return success dict with chunk count."""
    with patch("backend.rag_pipeline.extract_text") as mock_extract, \
         patch("backend.rag_pipeline.chunk_text") as mock_chunk, \
         patch("backend.rag_pipeline.vector_store") as mock_vs:

        mock_extract.return_value = "Sample document text " * 50
        mock_chunk.return_value = ["chunk1", "chunk2", "chunk3"]
        mock_vs.add_texts.return_value = None

        from backend.rag_pipeline import ingest_document
        result = ingest_document("fake/path.pdf", "test.pdf")

        assert result["status"] == "success"
        assert result["chunks_stored"] == 3
        assert result["document"] == "test.pdf"


def test_ingest_document_empty_text_raises():
    """ingest_document should raise ValueError for unreadable PDF."""
    with patch("backend.rag_pipeline.extract_text") as mock_extract:
        mock_extract.return_value = ""

        from backend.rag_pipeline import ingest_document
        with pytest.raises(ValueError):
            ingest_document("fake/path.pdf", "empty.pdf")


def test_query_db_returns_chunks():
    """query_db should return list of dicts with text and source."""
    mock_doc = MagicMock()
    mock_doc.page_content = "Sample chunk text"
    mock_doc.metadata = {"source": "test.pdf", "chunk_index": 0}

    with patch("backend.rag_pipeline.vector_store") as mock_vs:
        mock_vs.similarity_search_with_score.return_value = [
            (mock_doc, 0.85)
        ]

        from backend.rag_pipeline import query_db
        results = query_db("test question", k=1)

        assert len(results) == 1
        assert results[0]["text"] == "Sample chunk text"
        assert results[0]["source"] == "test.pdf"


def test_generate_answer_no_chunks_returns_fallback():
    """generate_answer should return fallback when no chunks retrieved."""
    with patch("backend.rag_pipeline.query_db") as mock_query:
        mock_query.return_value = []

        from backend.rag_pipeline import generate_answer
        result = generate_answer("unanswerable question")

        assert "don't have enough information" in result["answer"].lower()
        assert result["sources"] == []


def test_generate_answer_returns_answer_and_sources():
    """generate_answer should return answer and sources on success."""
    mock_chunks = [
        {"text": "AI is transforming industries.", "source": "test.pdf", "chunk_index": 0}
    ]
    mock_response = MagicMock()
    mock_response.content = "AI is transforming industries."

    with patch("backend.rag_pipeline.query_db") as mock_query, \
         patch("backend.rag_pipeline.llm") as mock_llm:

        mock_query.return_value = mock_chunks
        mock_llm.invoke.return_value = mock_response

        from backend.rag_pipeline import generate_answer
        result = generate_answer("What does the document say about AI?")

        assert result["answer"] == "AI is transforming industries."
        assert "test.pdf" in result["sources"]    