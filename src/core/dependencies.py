from functools import lru_cache
from src.vectorstore.chroma_store import ChromaStore

@lru_cache()
def get_vector_store():
    return ChromaStore()

@lru_cache()
def get_document_cache():
    return {}  # Simple in-memory cache for now 