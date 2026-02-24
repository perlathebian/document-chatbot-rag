import streamlit as st
import requests
import os

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page setup
st.set_page_config(
    page_title="Document Chatbot",
    page_icon="📄",
    layout="wide"
)

# Session state initialization
# These persist across Streamlit reruns within the same browser session
if "uploaded_doc" not in st.session_state:
    st.session_state.uploaded_doc = None      # name of the currently uploaded PDF
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None       # last answer returned by the API
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []        # sources for the last answer
if "last_question" not in st.session_state:
    st.session_state.last_question = None     # last question asked

# Sidebar
with st.sidebar:
    st.title("📄 Document Chatbot")
    st.markdown("---")

    st.subheader("What is RAG?")
    st.markdown("""
    **RAG (Retrieval-Augmented Generation)** is an AI technique that lets you 
    ask questions about your own documents.

    Instead of relying on the AI's training data, RAG works in two phases:

    1. **Ingestion**: Your PDF is split into small chunks of text. Each chunk 
    is converted into a numerical vector (embedding) that represents its meaning, 
    then stored in a vector database.

    2. **Retrieval**: When you ask a question, it's converted into the same kind 
    of vector. The system finds the chunks most similar in meaning to your question 
    and sends them to the AI as context.

    The AI reads only those chunks to generate its answer; grounding the response 
    in your actual documents rather than hallucinating from memory.
    """)

    st.markdown("---")
    st.subheader("Tech Stack")
    st.markdown("""
    - **Embeddings:** sentence-transformers
    - **Vector DB:** ChromaDB
    - **LLM:** Groq (Llama 3.3 70B)
    - **Backend:** FastAPI
    - **Frontend:** Streamlit
    """)

    # Show API connection status
    st.markdown("---")
    st.subheader("API Status")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if response.status_code == 200:
            st.success("👍 API Connected")
        else:
            st.error("👎 API Error")
    except requests.exceptions.ConnectionError:
        st.error("API Offline (start the FastAPI server)")
    except requests.exceptions.Timeout:
        st.error("⌚ API timeout")

# Main Area
st.title("Ask Questions About Your Documents!")
st.markdown("Upload a PDF, then ask any question about its contents.")
st.markdown("---")

# Section 1: Upload
st.subheader("1. Upload a PDF Document")

uploaded_file = st.file_uploader(
    label="Choose a PDF file",
    type=["pdf"],
    help="Upload a text-based PDF. Scanned image PDFs are not supported."
)

if uploaded_file is not None:
    # Only upload if it's a new file (not the same one already uploaded)
    if st.session_state.uploaded_doc != uploaded_file.name:
        with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()
                    st.session_state.uploaded_doc = uploaded_file.name
                    st.success(
                        f"☑️ '{data['document']}' ingested successfully — "
                        f"{data['chunks_stored']} chunks stored."
                    )
                else:
                    error = response.json().get("detail", "Unknown error")
                    st.error(f"❌ Upload failed: {error}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure the FastAPI server is running.")
            except requests.exceptions.Timeout:
                st.error("⌚ Upload timed out. The PDF may be too large.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
    else:
        st.info(f"☑️ '{uploaded_file.name}' is already ingested and ready to query!")

# Section 2: Ask a Question
st.markdown("---")
st.subheader("2. Ask a Question")

# Optional source filter (only show if a doc has been uploaded this session)
use_filter = False
if st.session_state.uploaded_doc:
    use_filter = st.checkbox(
        f"Search only in '{st.session_state.uploaded_doc}'",
        value=True,
        help="When checked, only searches the document uploaded above. "
             "Uncheck to search across all documents in the database."
    )

question = st.text_input(
    label="Your question",
    placeholder="e.g. What is the main argument of this paper?",
    help="Ask anything about the content of your uploaded PDF."
)

ask_button = st.button("Ask", type="primary", disabled=(not question.strip()))

if ask_button and question.strip():
    source_filter = st.session_state.uploaded_doc if use_filter else None

    with st.spinner("Searching documents and generating answer..."):
        try:
            payload = {"question": question.strip()}
            if source_filter:
                payload["source"] = source_filter

            response = requests.post(
                f"{API_BASE_URL}/query",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                st.session_state.last_answer = data["answer"]
                st.session_state.last_sources = data["sources"]
                st.session_state.last_question = question.strip()

            elif response.status_code == 400:
                error = response.json().get("detail", "Bad request")
                st.error(f"❌ {error}")
            else:
                st.error(f"❌ API error {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure the FastAPI server is running.")
        except requests.exceptions.Timeout:
            st.error("⌚ Request timed out. Try a shorter question.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

# Section 3: Display Answer
if st.session_state.last_answer:
    st.markdown("---")
    st.subheader("Answer")

    # Show the question that was asked
    st.markdown(f"**Q: {st.session_state.last_question}**")

    # Answer in a styled container
    with st.container(border=True):
        st.markdown(st.session_state.last_answer)

    # Sources
    if st.session_state.last_sources:
        st.markdown("**Sources:**")
        for source in st.session_state.last_sources:
            st.markdown(f"- 📄 {source}")
    else:
        st.markdown("*No sources found.*")