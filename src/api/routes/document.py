from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from src.agents.coordinator import CoordinatorAgent
from src.tools.document_tools import DocumentAnalysisTool, StructureDetectionTool
from src.extractors import TextExtractor, ImageExtractor, TableExtractor
from src.core.dependencies import get_vector_store, get_document_cache
from src.processors.parallel_processor import ParallelDocumentProcessor
from src.core.error_handling import ProcessingError
from src.core.initialization import get_extractors
from src.vectorstore.chroma_store import ChromaStore
from src.core.cache import DocumentCache
import aiofiles
import os
from loguru import logger
from typing import Dict, Any
from chromadb.api.models import Collection

router = APIRouter(prefix="/documents", tags=["documents"])

async def save_upload_file(upload_file: UploadFile) -> str:
    file_path = f"uploads/{upload_file.filename}"
    os.makedirs("uploads", exist_ok=True)
    
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    
    return file_path

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_cache: DocumentCache = Depends(get_document_cache)
):
    try:
        logger.info(f"Received upload request for file: {file.filename}")
        
        # Save uploaded file
        file_path = await save_upload_file(file)
        logger.debug(f"File saved to: {file_path}")
        
        # Generate cache key from file path
        cache_key = document_cache.get_cache_key(file_path)
        
        # Check cache first
        cached_result = document_cache.get(cache_key)
        if cached_result:
            logger.info(f"Returning cached result for {file_path}")
            return cached_result

        logger.info("Initializing document processing tools")
        tools = {
            "analysis": [
                DocumentAnalysisTool(),
                StructureDetectionTool()
            ],
            "extraction": get_extractors()
        }
        
        # Process document
        try:
            vector_store = ChromaStore()
            coordinator = CoordinatorAgent(tools, vector_store)
            processor = ParallelDocumentProcessor(coordinator)
            result = await processor.process_document(file_path)
            
            response = {
                "status": "completed",
                "document_id": cache_key,
                "content": result
            }
            
            # Cache the result
            document_cache.set(cache_key, response)
            return response
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            error_response = {
                "status": "error",
                "document_id": cache_key,
                "error": str(e)
            }
            document_cache.set(cache_key, error_response)
            raise HTTPException(status_code=500, detail=str(e))
            
    except Exception as e:
        logger.error(f"Error processing document upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}")
async def get_document(
    document_id: str,
    document_cache=Depends(get_document_cache)
):
    result = document_cache.get(document_id)
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
    return result 