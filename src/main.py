from src.extractors import TextExtractor, ImageExtractor, TableExtractor
from src.vectorstore.chroma_store import ChromaStore
from src.agents.orchestrator import OrchestratorAgent
from src.tools.document_tools import DocumentSearchTool, ContentExtractionTool
from src.rag.pipeline import RAGPipeline
from src.core.cache import DocumentCache
from src.core.error_handling import setup_error_handling, handle_extraction_error

async def process_document(document_path: str):
    # Initialize components
    extractors = {
        "text": TextExtractor(),
        "image": ImageExtractor(),
        "table": TableExtractor()
    }
    
    vector_store = ChromaStore()
    document_cache = DocumentCache()
    
    # Initialize tools
    tools = [
        DocumentSearchTool(vector_store),
        ContentExtractionTool(extractors)
    ]
    
    # Initialize orchestrator and RAG pipeline
    orchestrator = OrchestratorAgent(tools)
    rag_pipeline = RAGPipeline(vector_store)
    
    # Process document
    try:
        # Check cache first
        cached_result = document_cache.get(document_path)
        if cached_result:
            return cached_result
            
        # Extract content
        content = await orchestrator.process_query(f"Extract content from {document_path}")
        
        # Store in vector store
        vector_store.add_documents([content])
        
        # Cache the result
        document_cache.set(document_path, content)
        
        return content
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise ProcessingError(f"Failed to process document: {str(e)}")

if __name__ == "__main__":
    setup_error_handling()
    # Use the system
    result = await process_document("path/to/your/document.pdf") 