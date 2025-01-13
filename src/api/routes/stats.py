from fastapi import APIRouter, Depends
from src.api.models import CollectionStats
from src.core.dependencies import get_vector_store

router = APIRouter(prefix="/stats", tags=["stats"])

@router.get("/", response_model=CollectionStats)
async def get_stats(vector_store=Depends(get_vector_store)):
    stats = {
        "total_documents": len(vector_store.collection),
        "collections": ["default"],
        "embedding_dimensions": 384  # Default for MiniLM-L6-v2
    }
    return CollectionStats(**stats) 