from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from src.vectorstore.chroma_store import ChromaStore

class HybridRetriever:
    def __init__(self, vector_store: ChromaStore):
        self.vector_store = vector_store
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    async def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        # Get initial candidates using vector similarity
        initial_results = await self.vector_store.similarity_search(query, k=k*2)
        
        # Rerank using cross-encoder
        pairs = [(query, doc.page_content) for doc in initial_results]
        scores = self.cross_encoder.predict(pairs)
        
        # Combine results with scores
        scored_results = list(zip(initial_results, scores))
        reranked_results = sorted(scored_results, key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in reranked_results[:k]] 