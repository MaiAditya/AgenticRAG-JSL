from langchain.tools import BaseTool
from typing import Dict, Any, List, Optional
from src.vectorstore.chroma_store import ChromaStore
from loguru import logger
import io
from PIL import Image

class DocumentAnalysisTool(BaseTool):
    name = "document_analysis"
    description = "Analyze document structure and identify components"

    async def _run(self, page: Any) -> Dict[str, Any]:
        try:
            components = []
            
            # Extract text content
            text_content = page.get_text() if hasattr(page, 'get_text') else str(page)
            
            # Add text content
            if text_content:
                components.append({
                    "type": "text",
                    "content": text_content
                })
            
            # Detect tables directly from page
            tables = await self._detect_tables(page)
            if tables:
                logger.info(f"Detected {len(tables)} tables")
                for table in tables:
                    components.append({
                        "type": "table",
                        "content": table["cells"],
                        "structure": table["structure"],
                        "num_rows": table["num_rows"],
                        "num_cols": table["num_cols"]
                    })
            
            # Detect images directly from page
            images = self._detect_images(page)
            if images:
                logger.info(f"Detected {len(images)} images")
                for image in images:
                    components.append({
                        "type": "image",
                        "content": image["content"],
                        "metadata": image["metadata"]
                    })
            
            return {
                "type": "page",
                "content": text_content,
                "components": components
            }
        except Exception as e:
            logger.error(f"Error in document analysis: {str(e)}")
            return {"error": str(e)}

    async def _detect_tables(self, page: Any) -> List[Dict[str, Any]]:
        try:
            # Convert page to image for table detection
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            table_extractor = TableExtractor()
            result = await table_extractor.extract(image)
            
            if result and "tables" in result:
                logger.info("Successfully extracted tables from page")
                return result["tables"]
            return []
            
        except Exception as e:
            logger.error(f"Error detecting tables: {str(e)}")
            return []

    def _detect_images(self, page: Any) -> List[Dict[str, Any]]:
        try:
            # Get images directly from page object
            image_list = page.get_images()
            images = []
            
            for img in image_list:
                xref = img[0]
                image_info = {
                    "content": {
                        "xref": xref,
                        "width": img[2],
                        "height": img[3],
                        "data": page.extract_image(xref)
                    },
                    "metadata": {
                        "type": "image",
                        "format": img[1],
                        "colorspace": img[4]
                    }
                }
                images.append(image_info)
                logger.info(f"Extracted image: {img[1]} format, size: {img[2]}x{img[3]}")
            
            return images
            
        except Exception as e:
            logger.error(f"Error detecting images: {str(e)}")
            return []

    def _arun(self, document: str) -> Dict[str, Any]:
        return self._run(document)

class StructureDetectionTool(BaseTool):
    name = "structure_detection"
    description = "Detect and classify document structural elements"

    def _run(self, document: Any) -> Dict[str, Any]:
        try:
            # Extract text content from the page
            if hasattr(document, 'get_text'):
                text_content = document.get_text()
            else:
                text_content = str(document)

            # Basic structure detection
            structure = {
                "type": "page",
                "content": text_content,
                "components": [
                    {
                        "type": "text",
                        "content": text_content
                    }
                ]
            }
            return structure
        except Exception as e:
            logger.error(f"Error in structure detection: {str(e)}")
            return {
                "type": "error",
                "error": str(e)
            }

    async def _arun(self, document: Any) -> Dict[str, Any]:
        return self._run(document)

class ContentExtractionTool(BaseTool):
    name = "content_extraction"
    description = "Extract specific content from documents"

    def __init__(self, extractors: Dict[str, Any]):
        super().__init__()
        self.extractors = extractors

    async def _arun(self, content_type: str, content: Any) -> Dict[str, Any]:
        extractor = self.extractors.get(content_type)
        if not extractor:
            raise ValueError(f"No extractor found for content type: {content_type}")
        return await extractor.extract(content) 