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
import time

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
        os.makedirs("logs/table_detections/cells", exist_ok=True)

    async def extract(self, image) -> dict:
        try:
            # Convert PyMuPDF Pixmap to PIL Image
            if hasattr(image, 'tobytes'):
                logger.debug("Converting Pixmap to PIL Image")
                try:
                    # Convert Pixmap to numpy array directly
                    img_array = np.frombuffer(image.samples, dtype=np.uint8).reshape(
                        image.height, image.width, image.n
                    )
                    # Convert to RGB if needed
                    if image.n == 4:  # RGBA
                        img_array = img_array[:, :, :3]
                    image = Image.fromarray(img_array)
                    
                except Exception as e:
                    logger.error(f"Direct conversion failed: {str(e)}")
                    # Fallback to raw pixel data
                    try:
                        raw_data = image.tobytes("raw")
                        if image.n == 4:  # RGBA
                            mode = "RGBA"
                        else:
                            mode = "RGB"
                        image = Image.frombytes(mode, (image.width, image.height), raw_data)
                    except Exception as e2:
                        logger.error(f"Fallback conversion failed: {str(e2)}")
                        return None
                    
                logger.info(f"Successfully converted image to PIL Image with size {image.size}")

                # Save original image for debugging
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("logs/table_detections/originals", exist_ok=True)
                original_path = f"logs/table_detections/originals/original_{timestamp}.png"
                image.save(original_path, format='JPEG')  # Use JPEG instead of PNG
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
                threshold=0.7
            )[0]

            # Extract table regions and process with OCR
            cells = []
            table_image = np.array(image)
            
            # Convert to grayscale if needed
            if len(table_image.shape) == 3:
                table_image = cv2.cvtColor(table_image, cv2.COLOR_RGB2GRAY)

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i) for i in box.tolist()]
                x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]

                # Extract cell region
                cell = table_image[y:y+h, x:x+w]
                
                if cell.size == 0:
                    continue

                # Preprocess cell image
                cell_scaled = cv2.resize(cell, None, fx=2, fy=2)
                _, cell_scaled = cv2.threshold(cell_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                try:
                    # Optimize OCR settings for table text
                    custom_config = r'--oem 3 --psm 6'
                    text = pytesseract.image_to_string(
                        cell_scaled,
                        config=custom_config,
                        lang='eng'
                    ).strip()
                    
                    if text:
                        cells.append({
                            "bbox": [x, y, w, h],
                            "text": text,
                            "confidence": float(score),
                            "is_header": y < table_image.shape[0] * 0.2
                        })
                        
                        # Save cell image for debugging
                        os.makedirs("logs/table_detections/cells", exist_ok=True)
                        debug_path = f"logs/table_detections/cells/cell_{len(cells)}_{timestamp}.png"
                        cv2.imwrite(debug_path, cell_scaled)
                        logger.debug(f"Saved cell image to {debug_path}")
                        
                except Exception as e:
                    logger.error(f"OCR error: {str(e)}")
                    continue

            if cells:
                result = {
                    "table_data": {
                        "cells": cells,
                        "num_rows": len(set(c["bbox"][1] for c in cells)),
                        "num_cols": len(set(c["bbox"][0] for c in cells)),
                        "headers": [c["text"] for c in cells if c["is_header"]]
                    },
                    "metadata": {
                        "confidence_scores": [float(score) for score in results["scores"]],
                        "processing_time": time.time() - start_time if 'start_time' in locals() else None
                    }
                }
                logger.info(f"Successfully extracted table with {len(cells)} cells")
                return result
            
            logger.warning("No table cells detected in the image")
            return None
            
        except Exception as e:
            logger.error(f"Error in table extraction: {str(e)}")
            return None

    def _analyze_table_structure(self, table_image: Image.Image) -> Dict[str, Any]:
        try:
            # Convert and preprocess
            table_np = np.array(table_image)
            original = table_np.copy()
            
            # Better preprocessing for medical text
            gray = cv2.cvtColor(table_np, cv2.COLOR_RGB2GRAY)
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 21, 10
            )
            
            # Remove noise
            kernel = np.ones((2,2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Detect table structure
            scale = table_np.shape[1] // 30
            ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale))
            hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale, 1))
            
            # Detect lines
            vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ver_kernel, iterations=2)
            horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hor_kernel, iterations=2)
            
            # Find table grid
            grid = cv2.addWeighted(vertical, 1, horizontal, 1, 0)
            
            # Find cells with better contour detection
            contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process cells
            min_area = 500  # Larger minimum area
            cells = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                cell_roi = original[y:y+h, x:x+w]
                
                if cell_roi.size == 0:
                    continue
                    
                # Enhanced OCR preprocessing
                cell_gray = cv2.cvtColor(cell_roi, cv2.COLOR_RGB2GRAY)
                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cell_gray = clahe.apply(cell_gray)
                
                # Scale up image
                scale_factor = 3
                cell_scaled = cv2.resize(cell_gray, None, 
                                       fx=scale_factor, 
                                       fy=scale_factor, 
                                       interpolation=cv2.INTER_CUBIC)
                
                # Improve contrast
                cell_scaled = cv2.normalize(cell_scaled, None, 0, 255, cv2.NORM_MINMAX)
                
                try:
                    # Optimize OCR settings for medical text
                    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()/.+-%[] " -c tessedit_char_blacklist="|~`@#$^&*_={};<>?"'
                    text = pytesseract.image_to_string(
                        cell_scaled,
                        config=custom_config,
                        lang='eng'
                    ).strip()
                    
                    if text:
                        cells.append({
                            "bbox": [x, y, w, h],
                            "text": text,
                            "is_header": y < table_image.size[1] * 0.2
                        })
                        
                        # Save cell image for debugging
                        debug_path = f"logs/table_detections/cells/cell_{len(cells)}.png"
                        cv2.imwrite(debug_path, cell_scaled)
                        
                except Exception as e:
                    logger.error(f"OCR error: {str(e)}")
                    continue
            
            # Organize cells into structured data
            if cells:
                return {
                    "cells": cells,
                    "num_rows": len(set(c["bbox"][1] for c in cells)),
                    "num_cols": len(set(c["bbox"][0] for c in cells)),
                    "headers": [c["text"] for c in cells if c["is_header"]]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing table structure: {str(e)}")
            return None

    def _detect_lines(self, img: np.ndarray, direction: str, size: int) -> np.ndarray:
        """Detect horizontal or vertical lines in the image with improved parameters."""
        # Create structure element
        if direction == "horizontal":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
        
        # Apply morphology operations
        eroded = cv2.erode(img, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        
        return dilated 