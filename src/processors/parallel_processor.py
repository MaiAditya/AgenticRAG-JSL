from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import List, Dict, Any
from PIL import Image
import fitz  # PyMuPDF
from src.extractors.text_extractor import TextExtractor
from src.extractors.image_extractor import ImageExtractor
from src.extractors.table_extractor import TableExtractor

class ParallelDocumentProcessor:
    def __init__(self, max_workers: int = 4):
        self.text_extractor = TextExtractor()
        self.image_extractor = ImageExtractor()
        self.table_extractor = TableExtractor()
        self.max_workers = max_workers

    async def process_page(self, page) -> Dict[str, Any]:
        # Extract text
        text_content = await self.text_extractor.extract(page.get_text())
        
        # Extract images
        image_list = []
        for img_index, img in enumerate(page.get_images()):
            pix = fitz.Pixmap(page.parent, img[0])
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            image_content = await self.image_extractor.extract(image)
            image_list.append(image_content)

        # Extract tables
        tables = await self.table_extractor.extract(page)

        return {
            "page_num": page.number,
            "text": text_content,
            "images": image_list,
            "tables": tables
        }

    async def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        doc = fitz.open(file_path)
        
        # Process pages in parallel
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, self.process_page, page)
                for page in doc
            ]
            results = await asyncio.gather(*tasks)
        
        return results 