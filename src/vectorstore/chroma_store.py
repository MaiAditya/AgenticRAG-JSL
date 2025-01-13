import chromadb
from chromadb.config import Settings as ChromaSettings
from core.config import settings
from typing import List, Dict, Any

class ChromaStore:
    def __init__(self):
        self.client = chromadb.Client(ChromaSettings(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            anonymized_telemetry=False
        ))
        self.collection = self.client.create_collection(
            name="cpg_documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        self.collection.add(
            embeddings=embeddings,
            documents=[doc["content"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents]
        ) 