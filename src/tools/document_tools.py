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
            # Convert page to numpy array for better handling
            pix = page.get_pixmap()
            
            # Create table extractor instance
            table_extractor = TableExtractor()
            result = await table_extractor.extract(pix)
            
            if result and "table_data" in result and result["table_data"]:
                logger.info(f"Successfully extracted {len(result['table_data'])} tables")
                return [{
                    "cells": table_data,
                    "structure": table_data,
                    "num_rows": len(table_data),
                    "num_cols": len(table_data[0]) if table_data else 0,
                    "metadata": result["metadata"]
                } for table_data in result["table_data"]]
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

    async def _run(self, file_path: str) -> Dict[str, Any]:
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(file_path)
            total_pages = len(doc)
            processed_pages = 0
            components = []
            
            for page in doc:
                # Get text content
                text_content = page.get_text()
                if text_content.strip():
                    components.append({
                        "type": "text",
                        "content": text_content,
                        "page_number": page.number
                    })
                
                # Process tables
                pix = page.get_pixmap()
                if pix:
                    components.append({
                        "type": "table",
                        "image": pix,
                        "page_number": page.number,
                        "raw_image": pix.tobytes("png")
                    })
                
                # Process images
                image_list = page.get_images()
                for img in image_list:
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        if base_image:
                            image_data = base_image["image"]
                            components.append({
                                "type": "image",
                                "image": image_data,
                                "metadata": {
                                    "page_number": page.number,
                                    "width": img[2],
                                    "height": img[3],
                                    "colorspace": img[4],
                                    "xref": xref
                                }
                            })
                            logger.info(f"Added image from page {page.number} for processing")
                    except Exception as img_error:
                        logger.error(f"Error extracting image: {str(img_error)}")
                
                processed_pages += 1
            
            logger.info(f"Document structure detection completed: {len(components)} components found")
            return {
                "type": "document",
                "total_pages": total_pages,
                "processed_pages": processed_pages,
                "components": components
            }
        except Exception as e:
            logger.error(f"Error in structure detection: {str(e)}")
            raise

    async def _arun(self, file_path: str) -> Dict[str, Any]:
        return self._run(file_path)

    async def analyze(self, file_path: str) -> Dict[str, Any]:
        return await self._arun(file_path)

class ContentExtractionTool(BaseTool):
    name = "content_extraction"
    description = "Extract specific content from documents"

    def __init__(self, extractors: Dict[str, Any]):
        super().__init__()
        self.extractors = extractors

    async def _arun(self, content_type: str, content: Any) -> Dict[str, Any]:
        try:
            extractor = self.extractors.get(content_type)
            if not extractor:
                raise ValueError(f"No extractor found for content type: {content_type}")
            
            # Only pass the content argument to extract()
            return await extractor.extract(content)
        except Exception as e:
            logger.error(f"Error extracting component: {str(e)}")
            return {"error": str(e)} 