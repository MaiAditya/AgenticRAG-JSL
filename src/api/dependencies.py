from fastapi import Depends
from src.vectorstore.chroma_store import ChromaStore
from src.rag.pipeline import RAGPipeline
from src.core.cache import DocumentCache
from src.extractors.text_extractor import TextExtractor
from src.extractors.image_extractor import ImageExtractor
from src.extractors.table_extractor import TableExtractor

async def get_vector_store():
    return ChromaStore()

async def get_rag_pipeline(vector_store: ChromaStore = Depends(get_vector_store)):
    return RAGPipeline(vector_store)

async def get_document_cache():
    return DocumentCache()

async def get_extractors():
    return {
        "text": TextExtractor(),
        "image": ImageExtractor(),
        "table": TableExtractor()
    } 