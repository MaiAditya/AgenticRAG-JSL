from transformers import TableTransformerForObjectDetection, DetrFeatureExtractor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import numpy as np
from .base_extractor import BaseExtractor
from src.core.config import settings

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
        # Process cell image with TrOCR
        pixel_values = self.trocr_processor(cell_image, return_tensors="pt").pixel_values.to(settings.DEVICE)
        generated_ids = self.trocr_model.generate(pixel_values)
        return self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    async def extract(self, image: Image.Image) -> dict:
        # Detect and recognize table structure
        inputs = self.preprocess(image)
        outputs = self.model(**inputs)
        
        # Post-process outputs
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.feature_extractor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.7
        )[0]
        
        # Extract table structure and content
        tables = []
        cells = []
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            cell_box = box.tolist()
            cell_image = image.crop(cell_box)
            
            # Extract cell content using TrOCR (now synchronous)
            cell_content = self.extract_cell_content(cell_image)
            
            cells.append({
                "cell_type": self.model.config.id2label[label.item()],
                "confidence": score.item(),
                "bbox": cell_box,
                "content": cell_content,
                "structure_type": "header" if label.item() < 2 else "data"
            })
        
        # Organize cells into table structure
        if cells:
            table_structure = self._organize_table_structure(cells)
            tables.append({
                "cells": cells,
                "structure": table_structure,
                "num_rows": len(table_structure),
                "num_cols": len(table_structure[0]) if table_structure else 0
            })
        
        return {"tables": tables}

    def _organize_table_structure(self, cells):
        # Sort cells by vertical position first, then horizontal
        sorted_cells = sorted(cells, key=lambda x: (x["bbox"][1], x["bbox"][0]))
        
        # Group cells into rows based on vertical position
        rows = []
        current_row = []
        current_y = None
        
        for cell in sorted_cells:
            y_pos = cell["bbox"][1]
            if current_y is None or abs(y_pos - current_y) < 10:  # Threshold for same row
                current_row.append(cell)
                current_y = y_pos
            else:
                if current_row:
                    rows.append(sorted(current_row, key=lambda x: x["bbox"][0]))
                current_row = [cell]
                current_y = y_pos
                
        if current_row:
            rows.append(sorted(current_row, key=lambda x: x["bbox"][0]))
            
        return rows 