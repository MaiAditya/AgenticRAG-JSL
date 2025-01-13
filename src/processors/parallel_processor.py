from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import List, Dict, Any
from PIL import Image
import fitz  # PyMuPDF
from src.extractors.text_extractor import TextExtractor
from src.extractors.image_extractor import ImageExtractor
from src.extractors.table_extractor import TableExtractor
from src.agents.coordinator import CoordinatorAgent

class ParallelDocumentProcessor:
    def __init__(self, coordinator: CoordinatorAgent, max_workers: int = 4):
        self.coordinator = coordinator
        self.max_workers = max_workers

    async def process_document(self, file_path: str) -> Dict[str, Any]:
        doc = fitz.open(file_path)
        
        # Process pages in parallel through coordinator
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(self.coordinator.process_page(page))
                for page in doc
            ]
        
        results = [task.result() for task in tasks]
        return self.coordinator.combine_results(results) 