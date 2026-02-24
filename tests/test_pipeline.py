import pytest
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