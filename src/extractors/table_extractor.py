from transformers import TableTransformerForObjectDetection, DetrFeatureExtractor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import numpy as np
from .base_extractor import BaseExtractor
from src.core.config import settings
import io
import logging
import json
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class TableExtractor(BaseExtractor):
    def __init__(self):
        # Initialize TableNet for table structure recognition
        self.feature_extractor = DetrFeatureExtractor.from_pretrained('microsoft/table-transformer-structure-recognition')
        self.model = TableTransformerForObjectDetection.from_pretrained('microsoft/table-transformer-structure-recognition')
        
        # Initialize TrOCR for cell content extraction
        self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
        
        # Move models to device
        self.model.to(settings.DEVICE)
        self.trocr_model.to(settings.DEVICE)

    def preprocess(self, image: Image.Image) -> dict:
        return self.feature_extractor(images=image, return_tensors="pt")

    def extract_cell_content(self, cell_image: Image.Image) -> str:
        try:
            # Process cell image with TrOCR
            pixel_values = self.trocr_processor(cell_image, return_tensors="pt").pixel_values.to(settings.DEVICE)
            generated_ids = self.trocr_model.generate(pixel_values)
            text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting cell content: {str(e)}")
            return ""

    async def extract(self, image: Image.Image) -> dict:
        try:
            table_info = {
                "page_number": getattr(image, "page_number", None),
                "processing_start": datetime.now().isoformat()
            }
            
            logger.bind(table_extraction=True).info(
                f"Starting table extraction:\n{json.dumps(table_info, indent=2)}"
            )
            
            # Convert PyMuPDF Pixmap if needed
            if hasattr(image, 'tobytes'):
                img_data = image.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                table_info["image_size"] = image.size
            
            # Extract table structure and content
            table_data = await self._extract_table_structure(image)
            
            if table_data:
                table_info.update({
                    "num_rows": len(table_data),
                    "num_cols": len(table_data[0]) if table_data else 0,
                    "processing_end": datetime.now().isoformat()
                })
                
                logger.bind(table_extraction=True).info(
                    f"Extracted table data:\n{json.dumps(table_info, indent=2)}\n"
                    f"Table content:\n{json.dumps(table_data, indent=2)}"
                )
            
            return {
                "table_data": table_data,
                "metadata": table_info
            }
            
        except Exception as e:
            logger.error(f"Error in table extraction: {str(e)}")
            return {"error": str(e)}

    async def _extract_table_structure(self, image: Image.Image) -> List[List[Dict[str, Any]]]:
        try:
            # Prepare image for table detection
            inputs = self.preprocess(image)
            outputs = self.model(**inputs)
            
            # Post-process outputs
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.feature_extractor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.7
            )[0]
            
            cells = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = box.tolist()
                cell_image = image.crop(box)
                
                # Extract text from cell
                cell_text = self.extract_cell_content(cell_image)
                
                cells.append({
                    "bbox": box,
                    "confidence": score.item(),
                    "type": self.model.config.id2label[label.item()],
                    "content": cell_text
                })
                
                logger.bind(table_extraction=True).info(
                    f"Extracted cell: {json.dumps({'type': self.model.config.id2label[label.item()], 'content': cell_text}, indent=2)}"
                )
            
            # Organize cells into table structure
            return self._organize_table_structure(cells)
            
        except Exception as e:
            logger.error(f"Error extracting table structure: {str(e)}")
            return []

    def _organize_table_structure(self, cells: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        try:
            if not cells:
                return []
            
            # Sort cells by vertical position first, then horizontal
            sorted_cells = sorted(cells, key=lambda x: (x["bbox"][1], x["bbox"][0]))
            
            # Group cells into rows
            rows = []
            current_row = []
            current_y = sorted_cells[0]["bbox"][1]
            
            for cell in sorted_cells:
                if not current_row:
                    current_row.append(cell)
                    continue
                
                y_diff = abs(cell["bbox"][1] - current_y)
                if y_diff < 20:  # Cells in same row (adjust threshold as needed)
                    current_row.append(cell)
                else:
                    # Sort current row by x position
                    current_row.sort(key=lambda x: x["bbox"][0])
                    rows.append(current_row)
                    current_row = [cell]
                    current_y = cell["bbox"][1]
            
            # Add last row if exists
            if current_row:
                current_row.sort(key=lambda x: x["bbox"][0])
                rows.append(current_row)
            
            logger.bind(table_extraction=True).info(
                f"Organized table with {len(rows)} rows and {len(rows[0]) if rows else 0} columns"
            )
            
            return rows
            
        except Exception as e:
            logger.error(f"Error organizing table structure: {str(e)}")
            return [] 