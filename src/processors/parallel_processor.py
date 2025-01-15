from typing import List, Dict, Any
import fitz  # PyMuPDF
from src.core.types import DocumentProcessor
from loguru import logger
import asyncio

class ParallelDocumentProcessor:
    def __init__(self, coordinator: DocumentProcessor, max_workers: int = 4):
        self.coordinator = coordinator
        self.max_workers = max_workers

    async def process_document(self, file_path: str) -> Dict[str, Any]:
        try:
            # Process the entire document at once instead of page by page
            result = await self.coordinator.process_document(file_path)
            return result
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise ProcessingError(f"Failed to process document: {str(e)}") 