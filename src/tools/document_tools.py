from langchain.tools import BaseTool
from typing import Optional
from src.vectorstore.chroma_store import ChromaStore

class DocumentSearchTool(BaseTool):
    name = "document_search"
    description = "Search for relevant information in the document collection"

    def __init__(self, vector_store: ChromaStore):
        super().__init__()
        self.vector_store = vector_store

    def _run(self, query: str) -> str:
        results = self.vector_store.collection.query(
            query_texts=[query],
            n_results=3
        )
        return results

class ContentExtractionTool(BaseTool):
    name = "content_extraction"
    description = "Extract specific content from documents based on type (text, table, image)"

    def __init__(self, extractors: dict):
        super().__init__()
        self.extractors = extractors

    def _run(self, content_type: str, content: Any) -> dict:
        extractor = self.extractors.get(content_type)
        if not extractor:
            raise ValueError(f"No extractor found for content type: {content_type}")
        return extractor.extract(content) 