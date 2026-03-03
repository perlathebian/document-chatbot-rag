import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from backend.utils import get_chroma_client, extract_text, chunk_text

load_dotenv()

import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/rag_pipeline.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"
VECTORDB_PATH   = "db/vectordb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Loading embedding model 
# First run: downloads ~90MB model and caches locally
# Subsequent runs: loads from cache
logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},     # gpu not needed
    encode_kwargs={"normalize_embeddings": True}  # normalizes vectors for better cosine similarity
)
logger.info("Embedding model loaded")

# Initializing ChromaDB vector store
# langchain_chroma.Chroma wraps ChromaDB and connects it to the embedding model
# persist_directory: where ChromaDB writes its files on disk
logger.info(f"Initializing ChromaDB at: {VECTORDB_PATH}")
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=VECTORDB_PATH
)
logger.info("ChromaDB initialized")

# Initializing Groq LLM
# ChatGroq connects to Groq's API using the GROQ_API_KEY from .env
# llama3-8b-8192: fast, capable, 8192 token context window
# temperature=0: deterministic responses(same question = same answer)
logger.info("Initializing Groq LLM: llama-3.3-70b-versatile")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)
logger.info("Groq LLM initialized")

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
    logger.info(f"Ingestion started: {doc_name} ({file_path})")

    # Step 1: Extracting raw text from PDF
    text = extract_text(file_path)
    if not text:
        logger.error(f"Text extraction failed for: {doc_name}; may be image-based PDF")
        raise ValueError(f"Could not extract text from {file_path}. PDF may be image-based.")
    logger.info(f"Text extracted: {len(text)} chars from {doc_name}")

    # Step 2: Splitting into chunks
    chunks = chunk_text(text)
    logger.info(f"Text chunked: {len(chunks)} chunks created from {doc_name}")

    # Step 3: Building metadata for each chunk
    # Each chunk gets tagged with its source document and position
    metadatas = [{"source": doc_name, "chunk_index": i} for i in range(len(chunks))]

    # Step 4: Building unique ID for each chunk
    # ChromaDB requires a unique ID per document; we combine doc name & chunk index
    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]

    # Step 5: Adding to ChromaDB
    # LangChain's Chroma.add_texts() handles embedding & storage in one call
    try:
        vector_store.add_texts(
            texts=chunks,
            metadatas=metadatas,
            ids=ids
        )
    except Exception as e:
        logger.error(f"ChromaDB storage failed for {doc_name}: {e}")
        raise

    logger.info(f"Ingestion complete: {len(chunks)} chunks stored for {doc_name}")
    return {
        "document": doc_name,
        "chunks_stored": len(chunks),
        "status": "success"
    }

# Retrieval
def query_db(question: str, k: int = 4, source_filter: str = None) -> list[dict]:
    """
    Embeds the question and retrieves the top-k most similar chunks from ChromaDB.
    Optionally filters by source document name.

    Args:
        question: the user's question as plain text
        k: number of chunks to retrieve (default 4)
        source_filter: if provided, only searches chunks from this document

    Returns:
        list of dicts with chunk text and metadata
    """
    logger.info(f"Querying ChromaDB: '{question[:80]}' (k={k}, filter={source_filter})")

    # Build ChromaDB where filter if source is specified
    filter_dict = {"source": source_filter} if source_filter else None

    try:
        # similarity_search_with_score returns (Document, score) tuples
        results_with_scores = vector_store.similarity_search_with_score(
            query=question,
            k=k,
            filter=filter_dict
        )
    except Exception as e:
        logger.error(f"ChromaDB query failed: {e}")
        raise

    # Log retrieval scores for each chunk
    for i, (doc, score) in enumerate(results_with_scores):
        logger.info(
            f"Chunk {i+1}: source={doc.metadata.get('source', 'unknown')}, "
            f"chunk_index={doc.metadata.get('chunk_index', -1)}, "
            f"similarity_score={score:.4f}"
        )

    return [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index", -1)
        }
        for doc, score in results_with_scores
    ]


# Answer Generation 
def generate_answer(question: str, source_filter: str = None) -> dict:
    """
    Full RAG answer generation with optional source filtering:
    1. Retrieve relevant chunks from ChromaDB
    2. Build a prompt with those chunks as context
    3. Call LLM to generate answer
    4. Return answer + source document names

    Args:
        question: the user's question in plain text

    Returns:
        dict with "answer" (str) and "sources" (list of source filenames)
    """
    logger.info(f"generate_answer called: '{question[:80]}'")

    # Step 1: Retrieving relevant chunks
    # Pass source_filter through to query_db
    try:
        retrieved_chunks = query_db(question, k=4, source_filter=source_filter)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise

    if not retrieved_chunks:
        logger.warning("No chunks retrieved: returning fallback answer")
        return {
            "answer": "I don't have enough information to answer this question.",
            "sources": []
        }
    logger.info(f"Retrieved {len(retrieved_chunks)} chunks for answer generation")


    # Step 2: Building context string from retrieved chunks
    # Joining all chunk texts with a separator so the llm can read them distinctly
    context = "\n\n---\n\n".join([chunk["text"] for chunk in retrieved_chunks])

    # Step 3: Extracting unique source filenames for citations
    sources = list(set([chunk["source"] for chunk in retrieved_chunks]))

    # Step 4: Building the prompt
    # SystemMessage: sets the llm's behavior rules
    # HumanMessage: the actual question with context injected
    system_message = SystemMessage(content=(
        "You are a helpful assistant that answers questions based strictly on "
        "the provided document context. "
        "If the answer is not contained in the context, respond with: "
        "'I don't have enough information in the provided documents to answer this question.' "
        "Do not use any knowledge outside of the provided context. "
        "Be concise and accurate."
    ))

    human_message = HumanMessage(content=(
        f"Context from documents:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer based only on the context above:"
    ))

    # Step 5: Calling llm with timing
    logger.info("Calling Groq LLM...")
    
    llm_start = time.time()
    try:
        response = llm.invoke([system_message, human_message])
    except Exception as e:
        error_msg = str(e).lower()
        if "timeout" in error_msg or "timed out" in error_msg:
            logger.error(f"Groq API timeout: {e}")
            raise TimeoutError(f"LLM request timed out: {e}")
        elif "rate limit" in error_msg or "429" in error_msg:
            logger.error(f"Groq API rate limit: {e}")
            raise RuntimeError(f"LLM rate limit exceeded: {e}")
        else:
            logger.error(f"Groq API error: {e}")
            raise

    llm_elapsed = time.time() - llm_start
    logger.info(f"LLM response received in {llm_elapsed:.2f}s ({len(response.content)} chars)")


    return {
        "answer": response.content,
        "sources": sources
    }