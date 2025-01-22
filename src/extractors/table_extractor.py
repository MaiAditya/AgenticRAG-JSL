from transformers import DetrImageProcessor, TableTransformerForObjectDetection, Blip2Processor, Blip2ForConditionalGeneration
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

        # Initialize BLIP-2 for table description
        logger.info("Loading BLIP-2 model for table description...")
        self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        logger.info("BLIP-2 model loaded successfully")

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

            # Process detected tables with descriptions
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
                    
                    # Generate table description
                    description = await self._generate_table_description(table_region)
                    
                    # Extract table structure
                    structure = self._analyze_table_structure(table_region)
                    
                    if structure:
                        table_data = {
                            "bbox": bbox,
                            "confidence": score_val,
                            "description": description,
                            "structure": structure
                        }
                        tables.append(table_data)
                        
                        # Prepare data for vector store
                        vector_store_entry = {
                            "content": f"Table Description: {description}\n\nTable Content: {json.dumps(structure, indent=2)}",
                            "metadata": {
                                "type": "table",
                                "confidence": score_val,
                                "num_rows": structure["num_rows"],
                                "num_cols": structure["num_cols"],
                                "headers": structure["headers"]
                            }
                        }
                        
                        # Add to vector store through the coordinator
                        if hasattr(self, 'vector_store'):
                            await self.vector_store.add_texts(
                                texts=[vector_store_entry["content"]],
                                metadatas=[vector_store_entry["metadata"]]
                            )
                        
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

    async def _generate_table_description(self, table_image: Image.Image) -> str:
        """Generate a description of the table using BLIP-2"""
        try:
            # Prepare image for BLIP-2
            inputs = self.blip_processor(images=table_image, return_tensors="pt").to(self.device)
            
            # Generate description with specific prompt
            prompt = "Describe this table's content and structure in detail, including its columns and purpose if apparent."
            inputs['text'] = self.blip_processor(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.blip_model.generate(
                **inputs,
                max_length=150,
                num_beams=5,
                min_length=30,
                top_p=0.9,
                repetition_penalty=1.5
            )
            
            description = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            return description
            
        except Exception as e:
            logger.error(f"Error generating table description: {str(e)}")
            return "Error generating table description" 