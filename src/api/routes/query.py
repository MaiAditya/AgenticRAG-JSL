from fastapi import APIRouter, Depends, HTTPException
from src.api.models import QueryRequest, QueryResponse
from src.rag.advanced_pipeline import AdvancedRAGPipeline
from src.retrieval.hybrid_retriever import HybridRetriever
from src.core.dependencies import get_vector_store
from loguru import logger

router = APIRouter(prefix="/query", tags=["query"])

@router.post("/", response_model=QueryResponse)
async def query_documents(
    query_request: QueryRequest,
    vector_store=Depends(get_vector_store)
):
    try:
        logger.info(f"Received query request: {query_request.query}")
        
        retriever = HybridRetriever(vector_store)
        pipeline = AdvancedRAGPipeline(retriever)
        
        result = await pipeline.get_answer(query_request.query)
        
        return QueryResponse(
            answer=result["answer"],
            sources=[str(source) for source in result["sources"]],
            confidence=max(result.get("confidence", 0.8), 0.0),
            reasoning_chain=result["reasoning_chain"]
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 