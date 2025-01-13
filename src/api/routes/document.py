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
    file: UploadFile = File(...),
    vector_store: ChromaStore = Depends(get_vector_store),
    document_cache: DocumentCache = Depends(get_document_cache)
) -> Dict[str, Any]:
    try:
        logger.info(f"Received upload request for file: {file.filename}")
        
        # Save uploaded file
        file_path = await save_upload_file(file)
        logger.debug(f"File saved to: {file_path}")
        
        # Get singleton extractors
        extractors = get_extractors()
        
        # Initialize tools
        tools = [
            DocumentAnalysisTool(),
            StructureDetectionTool()
        ]
        
        # Initialize coordinator agent
        coordinator = CoordinatorAgent(
            extractors=extractors,
            tools=tools,
            vector_store=vector_store,
            document_cache=document_cache
        )
        
        # Process document
        result = await coordinator.process_document(file_path)
        return result
        
    except Exception as e:
        logger.error(f"Error processing document upload: {str(e)}")
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