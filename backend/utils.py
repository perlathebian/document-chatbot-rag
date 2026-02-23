import chromadb

def get_chroma_client():
    client = chromadb.PersistentClient(path="db/vectordb")
    return client
