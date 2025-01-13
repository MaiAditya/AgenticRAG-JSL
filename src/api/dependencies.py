from fastapi import Depends
from src.vectorstore.chroma_store import ChromaStore
from src.core.cache import DocumentCache
from functools import lru_cache

@lru_cache()
def get_vector_store():
    return ChromaStore()

@lru_cache()
def get_document_cache():
    return DocumentCache() 