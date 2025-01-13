from fastapi import APIRouter, Depends, HTTPException
from src.api.models import CollectionStats
from ..dependencies import get_vector_store

router = APIRouter(prefix="/stats", tags=["stats"])

@router.get("/", response_model=CollectionStats)
async def get_collection_stats(
    vector_store=Depends(get_vector_store)
):
    try:
        stats = vector_store.get_stats()
        return CollectionStats(
            total_documents=stats["total_documents"],
            collections=stats["collections"],
            embedding_dimensions=stats["embedding_dimensions"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 