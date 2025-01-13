import chromadb
from chromadb.config import Settings as ChromaSettings
from src.core.config import settings
from typing import List, Dict, Any

class ChromaStore:
    def __init__(self):
        self.client = chromadb.Client(ChromaSettings(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            anonymized_telemetry=False
        ))
        # Create collection with a unique name
        try:
            self.collection = self.client.get_collection("cpg_documents")
        except ValueError:
            self.collection = self.client.create_collection(
                name="cpg_documents",
                metadata={"hnsw:space": "cosine"}
            )
    
    def __getstate__(self):
        """Custom serialization method"""
        state = self.__dict__.copy()
        # Don't pickle the client and collection
        state['client'] = None
        state['collection'] = None
        return state

    def __setstate__(self, state):
        """Custom deserialization method"""
        self.__dict__.update(state)
        # Reinitialize the client and collection
        self.__init__()

    async def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        # Add documents to the collection
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=[f"doc_{i}" for i in range(len(texts))]
        ) 