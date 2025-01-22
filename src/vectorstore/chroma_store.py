import chromadb
from chromadb.config import Settings as ChromaSettings
from src.core.config import settings
from typing import List, Dict, Any
from langchain.schema import Document
from loguru import logger
from datetime import datetime

class ChromaStore:
    def __init__(self):
        self.client = chromadb.Client(ChromaSettings(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            anonymized_telemetry=False
        ))
        
        # Initialize embedding function
        self.embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction()
        
        # Create collection with a unique name
        try:
            self.collection = self.client.get_collection(
                name="cpg_documents",
                embedding_function=self.embedding_function
            )
        except ValueError:
            self.collection = self.client.create_collection(
                name="cpg_documents",
                embedding_function=self.embedding_function,
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

    async def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add texts to the collection with optional metadata"""
        try:
            if not texts:
                logger.warning("No texts provided to add to collection")
                return
            
            logger.info(f"Adding {len(texts)} texts to collection")
            
            # Validate and log text content
            for idx, text in enumerate(texts):
                logger.debug(f"Text {idx+1} length: {len(text)} characters")
                logger.debug(f"Text {idx+1} sample: {text[:100]}...")
                
                if metadatas and idx < len(metadatas):
                    logger.debug(f"Metadata {idx+1} keys: {list(metadatas[idx].keys())}")
            
            # Ensure we have metadata for each text
            if not metadatas:
                logger.warning("No metadata provided, creating empty metadata")
                metadatas = [{} for _ in texts]
            
            # Generate IDs for new documents
            timestamp = datetime.now().timestamp()
            ids = [f"doc_{i}_{timestamp}" for i in range(len(texts))]
            
            # Add documents to the collection
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(texts)} documents to collection")
            logger.info(f"New collection count: {self.collection.count()}")
            logger.debug(f"Added document IDs: {ids}")
            
        except Exception as e:
            logger.error(f"Error adding texts to collection: {str(e)}", exc_info=True)
            raise

    async def similarity_search(self, query: str, k: int = 10):
        """Perform similarity search using the Chroma collection"""
        try:
            # Check if collection is empty
            collection_count = self.collection.count()
            if collection_count == 0:
                logger.warning("Collection is empty. No documents to search.")
                return []

            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=min(k, collection_count)
            )
            
            # Convert results to document format
            documents = []
            if results and 'documents' in results and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    doc = results['documents'][0][i]
                    metadata = results['metadatas'][0][i] if results.get('metadatas') and len(results['metadatas'][0]) > i else {}
                    documents.append(Document(
                        page_content=doc,
                        metadata=metadata
                    ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return [] 

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": "cpg_documents",
                "embedding_dimensions": 384
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                "total_documents": 0,
                "collection_name": "cpg_documents",
                "embedding_dimensions": 384
            } 