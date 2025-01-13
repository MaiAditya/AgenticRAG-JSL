from typing import List, Dict, Any
import fitz  # PyMuPDF
from src.agents.coordinator import CoordinatorAgent
from loguru import logger
import asyncio

class ParallelDocumentProcessor:
    def __init__(self, coordinator: CoordinatorAgent, max_workers: int = 4):
        self.coordinator = coordinator
        self.max_workers = max_workers

    async def process_document(self, file_path: str) -> Dict[str, Any]:
        try:
            doc = fitz.open(file_path)
            
            # Create tasks for parallel processing
            tasks = []
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def process_with_semaphore(page):
                async with semaphore:
                    return await self.coordinator.process_page(page)
            
            # Create tasks for all pages
            for page in doc:
                tasks.append(process_with_semaphore(page))
            
            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks)
            
            doc.close()
            return self.coordinator.combine_results(results)
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise 