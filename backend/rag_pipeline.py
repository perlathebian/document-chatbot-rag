import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
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


# Initializing Groq LLM
# ChatGroq connects to Groq's API using the GROQ_API_KEY from .env
# llama3-8b-8192: fast, capable, 8192 token context window
# temperature=0: deterministic responses(same question = same answer)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
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


# Answer Generation 
def generate_answer(question: str) -> dict:
    """
    Full RAG answer generation:
    1. Retrieve relevant chunks from ChromaDB
    2. Build a prompt with those chunks as context
    3. Call LLM to generate answer
    4. Return answer + source document names

    Args:
        question: the user's question in plain text

    Returns:
        dict with "answer" (str) and "sources" (list of source filenames)
    """
    # Step 1: Retrieving relevant chunks
    retrieved_chunks = query_db(question, k=8)

    if not retrieved_chunks:
        return {
            "answer": "I don't have enough information to answer this question.",
            "sources": []
        }

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

    # Step 5: Calling the Groq model
    response = llm.invoke([system_message, human_message])

    return {
        "answer": response.content,
        "sources": sources
    }