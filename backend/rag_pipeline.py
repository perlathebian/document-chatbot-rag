import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from backend.utils import get_chroma_client, extract_text, chunk_text

load_dotenv()

COLLECTION_NAME = "documents"
VECTORDB_PATH   = "db/vectordb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Loading embedding model 
# First run: downloads ~90MB model and caches locally
# Subsequent runs: loads from cache
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},     # gpu not needed
    encode_kwargs={"normalize_embeddings": True}  # normalizes vectors for better cosine similarity
)

# Initializing ChromaDB vector store
# langchain_chroma.Chroma wraps ChromaDB and connects it to the embedding model
# persist_directory: where ChromaDB writes its files on disk
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=VECTORDB_PATH
)


# Ingestion
def ingest_document(file_path: str, doc_name: str) -> dict:
    """
    Full ingestion pipeline: PDF -> text -> chunks -> embeddings -> ChromaDB

    Args:
        file_path: absolute or relative path to the PDF file
        doc_name: readable name stored in metadata (filename)

    Returns:
        dict with ingestion summary
    """
    # Step 1: Extracting raw text from PDF
    text = extract_text(file_path)
    if not text:
        raise ValueError(f"Could not extract text from {file_path}. PDF may be image-based.")

    # Step 2: Splitting into chunks
    chunks = chunk_text(text)

    # Step 3: Building metadata for each chunk
    # Each chunk gets tagged with its source document and position
    metadatas = [{"source": doc_name, "chunk_index": i} for i in range(len(chunks))]

    # Step 4: Building unique ID for each chunk
    # ChromaDB requires a unique ID per document; we combine doc name & chunk index
    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]

    # Step 5: Adding to ChromaDB
    # LangChain's Chroma.add_texts() handles embedding & storage in one call
    vector_store.add_texts(
        texts=chunks,
        metadatas=metadatas,
        ids=ids
    )

    return {
        "document": doc_name,
        "chunks_stored": len(chunks),
        "status": "success"
    }


# Retrieval
def query_db(question: str, k: int = 4) -> list[dict]:
    """
    Embeds the question and retrieves the top-k most similar chunks from ChromaDB.
    Does not call the LLM (pure retrieval only)

    Args:
        question: the user's question as plain text
        k: number of chunks to retrieve (default 4)

    Returns:
        list of dicts, each containing the chunk text and its metadata
    """
    results = vector_store.similarity_search(
        query=question,
        k=k
    )

    # similarity_search returns LangChain Document objects
    # will convert them to plain dicts for easier handling later
    return [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index", -1)
        }
        for doc in results
    ]