from transformers import DetrImageProcessor, TableTransformerForObjectDetection
from PIL import Image
import torch
import numpy as np
from loguru import logger
import datetime
import os
import cv2
from typing import List, Dict, Any, Tuple
import io
import json
import pytesseract

class TableExtractor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing TableExtractor on {self.device}")
        
        # Initialize Table Transformer for table detection
        model_name = "microsoft/table-transformer-detection"
        self.processor = DetrImageProcessor.from_pretrained(
            model_name,
            do_resize=True,
            size={'shortest_edge': 800, 'longest_edge': 1333},
        )
        self.model = TableTransformerForObjectDetection.from_pretrained(model_name).to(self.device)
        self.threshold = 0.5

        # Create output directories for debugging
        os.makedirs("logs/table_detections/originals", exist_ok=True)
        os.makedirs("logs/table_detections/visualizations", exist_ok=True)

    async def extract(self, image) -> dict:
        try:
            # Convert PyMuPDF Pixmap to PIL Image
            if hasattr(image, 'tobytes'):
                logger.debug("Converting Pixmap to PIL Image")
                try:
                    # First try PPM format conversion
                    img_data = image.tobytes("ppm")
                    image = Image.open(io.BytesIO(img_data))
                except Exception as e:
                    logger.debug(f"PPM conversion failed, trying raw samples: {str(e)}")
                    # Fallback to raw samples method
                    samples = image.samples
                    mode = {
                        1: "L",
                        3: "RGB",
                        4: "RGBA"
                    }.get(image.n, "RGB")
                    image = Image.frombytes(mode, [image.width, image.height], samples)
                
                logger.bind(table_extraction=True).info(
                    f"Successfully converted image to PIL Image with size {image.size}"
                )

                # Save original image for debugging
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                original_path = f"logs/table_detections/originals/original_{timestamp}.png"
                image.save(original_path)
                logger.debug(f"Saved original image to {original_path}")

            # Prepare image for detection
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process outputs
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self.threshold
            )[0]

            # Process detected tables
            tables = []
            vis_image = np.array(image.copy())

            for idx, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
                score_val = score.item()
                if score_val > self.threshold and label.item() == 0:  # Table class
                    bbox = box.cpu().tolist()
                    x0, y0, x1, y1 = [int(coord) for coord in bbox]
                    
                    # Draw rectangle for visualization
                    cv2.rectangle(vis_image, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    cv2.putText(vis_image, f"Table {idx}: {score_val:.2f}", 
                              (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Extract table region
                    table_region = image.crop((x0, y0, x1, y1))
                    structure = self._analyze_table_structure(table_region)
                    
                    if structure:
                        table_data = {
                            "bbox": bbox,
                            "confidence": score_val,
                            "structure": structure
                        }
                        tables.append(table_data)
                        # Log each detected table's data
                        logger.bind(table_extraction=True).info(
                            f"Detected table {idx}:\n{json.dumps(table_data, indent=2)}"
                        )

            # Save visualization if tables were found
            if tables:
                vis_path = f"logs/table_detections/visualizations/detection_{timestamp}.png"
                cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                logger.info(f"Saved table detection visualization to {vis_path}")

                result = {
                    "table_data": [t["structure"] for t in tables],
                    "metadata": {
                        "num_tables": len(tables),
                        "confidences": [t["confidence"] for t in tables],
                        "processing_time": datetime.datetime.now().isoformat(),
                        "visualization_path": vis_path
                    }
                }
                # Log final extracted data
                logger.bind(table_extraction=True).info(
                    f"Final extracted table data:\n{json.dumps(result, indent=2)}"
                )
                return result

            logger.bind(table_extraction=True).info("No tables detected in image")
            return {"table_data": [], "metadata": {"num_tables": 0}}

        except Exception as e:
            logger.error(f"Error in table extraction: {str(e)}")
            return {"error": str(e)}

    def _analyze_table_structure(self, table_image: Image.Image) -> Dict[str, Any]:
        """Analyze table structure and extract text from cells using OCR."""
        try:
            # Convert to numpy array
            table_np = np.array(table_image)
            
            # Process table structure
            gray = cv2.cvtColor(table_np, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            
            # Detect lines
            horizontal = self._detect_lines(binary, "horizontal")
            vertical = self._detect_lines(binary, "vertical")
            
            # Find cell intersections
            intersections = cv2.bitwise_and(horizontal, vertical)
            
            # Get cell coordinates
            contours, _ = cv2.findContours(
                intersections,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Initialize Tesseract OCR
            import pytesseract
            
            cells = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract cell region from original image
                cell_roi = table_np[y:y+h, x:x+w]
                
                # Convert to grayscale for better OCR
                cell_gray = cv2.cvtColor(cell_roi, cv2.COLOR_RGB2GRAY)
                
                # Apply thresholding to clean up the image
                _, cell_binary = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Extract text using OCR
                try:
                    text = pytesseract.image_to_string(cell_binary, config='--psm 6').strip()
                except Exception as ocr_error:
                    logger.error(f"OCR error for cell at {x},{y}: {str(ocr_error)}")
                    text = ""
                
                cells.append({
                    "bbox": [x, y, x+w, y+h],
                    "text": text
                })
                
                # Log extracted text for debugging
                if text:
                    logger.debug(f"Extracted text from cell at {x},{y}: {text}")
            
            # Sort cells by position
            cells.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))  # Sort by y, then x
            
            # Group cells into rows based on y-coordinate similarity
            row_threshold = 10  # pixels
            rows = []
            current_row = []
            last_y = -row_threshold
            
            for cell in cells:
                y = cell["bbox"][1]
                if abs(y - last_y) > row_threshold and current_row:
                    rows.append(current_row)
                    current_row = []
                current_row.append(cell)
                last_y = y
            
            if current_row:
                rows.append(current_row)
            
            # Sort cells within each row by x-coordinate
            for row in rows:
                row.sort(key=lambda c: c["bbox"][0])
            
            # Flatten back to list while maintaining order
            cells = [cell for row in rows for cell in row]
            
            return {
                "cells": cells,
                "num_rows": len(rows),
                "num_cols": max(len(row) for row in rows) if rows else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing table structure: {str(e)}")
            return None

    def _detect_lines(self, img: np.ndarray, direction: str) -> np.ndarray:
        """Detect horizontal or vertical lines in the image."""
        kernel_length = img.shape[1]//40 if direction == "horizontal" else img.shape[0]//40
        kernel = np.ones((1, kernel_length)) if direction == "horizontal" else np.ones((kernel_length, 1))
        morphed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return morphed 