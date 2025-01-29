from functools import lru_cache
from src.vectorstore.chroma_store import ChromaStore
from src.core.cache import DocumentCache

@lru_cache()
def get_vector_store():
    """Get or create a singleton instance of ChromaStore"""
    return ChromaStore()

@lru_cache()
def get_document_cache():
    """Get or create a singleton instance of DocumentCache"""
    return DocumentCache() 