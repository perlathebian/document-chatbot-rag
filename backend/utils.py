import os
import chromadb
import pdfplumber
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()


def get_chroma_client():
    client = chromadb.PersistentClient(path="db/vectordb")
    return client


def extract_text(file_path: str) -> str:
    """
    Opens a PDF file and extracts all text from every page.
    Returns one big string with all pages joined by newlines.
    """
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def chunk_text(text: str) -> list[str]:
    """
    Splits a large text string into smaller overlapping chunks.
    chunk_size=500: each chunk is max 500 characters
    chunk_overlap=50: last 50 chars of each chunk repeat in the next to not lose context
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]   # tries these split points in order (recursive)
    )
    chunks = splitter.split_text(text)
    return chunks