from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    collection_name: Optional[str] = "default"

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

class DocumentResponse(BaseModel):
    document_id: str
    status: str
    processed_content: dict

class CollectionStats(BaseModel):
    total_documents: int
    collections: List[str]
    embedding_dimensions: int 