from fastapi import APIRouter, Depends
from src.api.models import CollectionStats
from src.core.dependencies import get_vector_store

router = APIRouter(prefix="/stats", tags=["stats"])

@router.get("/", response_model=CollectionStats)
async def get_stats(vector_store=Depends(get_vector_store)):
    stats = await vector_store.get_stats()
    return CollectionStats(
        total_documents=stats["total_documents"],
        collections=[stats["collection_name"]],
        embedding_dimensions=stats["embedding_dimensions"]
    ) 