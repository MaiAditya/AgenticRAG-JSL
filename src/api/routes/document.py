from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from src.agents.coordinator import CoordinatorAgent
from src.tools.document_tools import DocumentAnalysisTool, StructureDetectionTool
from src.extractors import TextExtractor, ImageExtractor, TableExtractor
from src.core.dependencies import get_vector_store, get_document_cache
from src.processors.parallel_processor import ParallelDocumentProcessor
from src.core.error_handling import ProcessingError
import aiofiles
import os

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
    vector_store=Depends(get_vector_store),
    document_cache=Depends(get_document_cache)
):
    try:
        # Save uploaded file
        file_path = await save_upload_file(file)
        
        # Check cache first
        cached_result = document_cache.get(file_path)
        if cached_result:
            return cached_result

        # Initialize tools and coordinator
        tools = {
            "analysis": [
                DocumentAnalysisTool(),
                StructureDetectionTool()
            ],
            "extraction": {
                "text": TextExtractor(),
                "image": ImageExtractor(),
                "table": TableExtractor()
            }
        }
        
        coordinator = CoordinatorAgent(tools, vector_store)
        processor = ParallelDocumentProcessor(coordinator)
        
        # Process document in background
        background_tasks.add_task(
            processor.process_document,
            file_path
        )
        
        return {
            "status": "processing",
            "document_id": file_path,
            "message": "Document upload successful. Processing started."
        }
        
    except Exception as e:
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