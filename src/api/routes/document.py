from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import List
import aiofiles
import os
from src.api.models import DocumentResponse
from src.core.error_handling import ProcessingError
from ..dependencies import get_vector_store, get_document_cache, get_extractors
from src.processors.parallel_processor import ParallelDocumentProcessor

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    vector_store=Depends(get_vector_store),
    document_cache=Depends(get_document_cache)
):
    try:
        processor = ParallelDocumentProcessor()
        
        # Process document in parallel
        results = await processor.process_document(file.filename)
        
        # Store results in vector store
        await vector_store.add_documents(results)
        
        return {
            "status": "success",
            "pages_processed": len(results),
            "details": {
                "text_chunks": sum(len(page["text"]) for page in results),
                "images_processed": sum(len(page["images"]) for page in results),
                "tables_extracted": sum(len(page["tables"]) for page in results)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    vector_store=Depends(get_vector_store)
):
    try:
        vector_store.delete_document(document_id)
        return {"status": "success", "message": f"Document {document_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 