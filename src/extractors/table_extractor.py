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

        # Initialize BLIP-2 with larger model for better descriptions
        logger.info("Loading BLIP-2 model for table description...")
        self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-6.7b",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
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

            # Process detected tables
            tables = []
            vis_image = np.array(image.copy())

            for idx, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
                score_val = score.item()
                if score_val > self.threshold and label.item() == 0:  # Table class
                    bbox = box.cpu().tolist()
                    x0, y0, x1, y1 = [int(coord) for coord in bbox]
                    
                    # Extract table region
                    table_region = image.crop((x0, y0, x1, y1))
                    
                    # First analyze table structure
                    structure = self._analyze_table_structure(table_region)
                    
                    if structure:
                        # Generate description only if structure is valid
                        description = await self._generate_table_description(table_region)
                        
                        table_data = {
                            "bbox": bbox,
                            "confidence": score_val,
                            "description": description,
                            "structure": structure
                        }
                        tables.append(table_data)
                        
                        # Draw visualization
                        cv2.rectangle(vis_image, (x0, y0), (x1, y1), (0, 255, 0), 2)
                        cv2.putText(vis_image, f"Table {idx}: {score_val:.2f}", 
                                  (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Log detection
                        logger.bind(table_extraction=True).info(
                            f"Detected table {idx}:\n{json.dumps(table_data, indent=2)}"
                        )

            # Save visualization and return results
            if tables:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                vis_path = f"logs/table_detections/visualizations/detection_{timestamp}.png"
                cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                
                result = {
                    "table_data": tables,  # Return full table data including descriptions
                    "metadata": {
                        "num_tables": len(tables),
                        "confidences": [t["confidence"] for t in tables],
                        "processing_time": datetime.datetime.now().isoformat(),
                        "visualization_path": vis_path
                    }
                }
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
        """Generate a detailed description of the table using BLIP-2"""
        try:
            if not isinstance(table_image, Image.Image):
                logger.error(f"Invalid image type: {type(table_image)}")
                return "Error: Invalid image format"

            # Prepare image for BLIP-2
            inputs = self.blip_processor(images=table_image, return_tensors="pt").to(self.device)
            
            # Carefully crafted prompt for medical table analysis
            prompt = """You are a medical document analyzer. Examine this table carefully and provide a detailed description including:

            1. Purpose: What is the main purpose or topic of this table?
            2. Structure: How is the information organized (columns, sections, hierarchy)?
            3. Content: What specific medical information, measurements, or categories are presented?
            4. Key Findings: What are the most important data points or patterns?
            5. Medical Context: How does this information relate to clinical practice or medical knowledge?
            6. Special Notes: Are there any warnings, contraindications, or critical values highlighted?

            Focus on medical terminology and clinical relevance. Be precise and concise."""
            
            text_inputs = self.blip_processor(text=prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.blip_model.generate(
                    **inputs,
                    max_length=500,  # Allow for longer, more detailed descriptions
                    min_length=200,  # Ensure comprehensive description
                    num_beams=5,
                    temperature=0.7,  # Slightly increase creativity while maintaining accuracy
                    top_p=0.9,
                    repetition_penalty=2.0,  # Increased to avoid repetition
                    length_penalty=1.5,  # Encourage detailed descriptions
                    do_sample=True  # Enable sampling for more natural text
                )
            
            description = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            
            # Add structural analysis
            structure_info = self._analyze_table_structure(table_image)
            content_summary = self._summarize_table_content(table_image)
            
            full_description = f"""
            Medical Table Analysis:
            ----------------------
            {description}

            Technical Details:
            ----------------
            {structure_info}

            Content Overview:
            ---------------
            {content_summary}
            """
            
            logger.debug(f"Generated detailed table description: {full_description}")
            return full_description
            
        except Exception as e:
            logger.error(f"Error generating table description: {str(e)}")
            return "Error generating table description"

    def _summarize_table_content(self, table_image: Image.Image) -> str:
        """Analyze and summarize table content patterns"""
        try:
            # Convert to numpy array for analysis
            table_np = np.array(table_image)
            
            # Basic image analysis for content patterns
            gray = cv2.cvtColor(table_np, cv2.COLOR_RGB2GRAY)
            text_density = np.mean(gray < 128)  # Estimate text density
            
            # Detect if table has header row
            top_region = gray[:int(gray.shape[0] * 0.2), :]
            has_header = np.mean(top_region < 128) > text_density
            
            # Analyze cell density and distribution
            cell_analysis = self._analyze_cell_distribution(gray)
            
            return f"""
            Content Density: {'High' if text_density > 0.3 else 'Medium' if text_density > 0.1 else 'Low'}
            Header Present: {'Yes' if has_header else 'No'}
            Cell Distribution: {cell_analysis}
            """
        except Exception as e:
            logger.error(f"Error in content summary: {str(e)}")
            return "Unable to analyze content"

    def _analyze_cell_distribution(self, gray_image: np.ndarray) -> str:
        """Analyze the distribution of cells in the table"""
        try:
            # Use adaptive thresholding to detect cells
            binary = cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 21, 10
            )
            
            # Analyze row and column patterns
            row_density = np.mean(binary, axis=1)
            col_density = np.mean(binary, axis=0)
            
            # Determine table characteristics
            regular_rows = np.std(row_density) < np.mean(row_density) * 0.5
            regular_cols = np.std(col_density) < np.mean(col_density) * 0.5
            
            return f"""
            Pattern: {'Regular grid' if regular_rows and regular_cols else 'Irregular layout'}
            Row Consistency: {'Uniform' if regular_rows else 'Variable'}
            Column Consistency: {'Uniform' if regular_cols else 'Variable'}
            """
        except Exception as e:
            logger.error(f"Error in cell distribution analysis: {str(e)}")
            return "Unable to analyze cell distribution"

    def _get_structure_description(self, table_image: Image.Image) -> str:
        """Generate description of table structure"""
        try:
            # Get basic dimensions
            width, height = table_image.size
            
            # Convert to grayscale for analysis
            gray_image = np.array(table_image.convert('L'))
            
            # Detect lines
            horizontal_lines = self._detect_lines(gray_image, "horizontal", width // 30)
            vertical_lines = self._detect_lines(gray_image, "vertical", height // 30)
            
            # Count approximate rows and columns
            row_positions = np.where(np.sum(horizontal_lines, axis=1) > width * 0.5)[0]
            col_positions = np.where(np.sum(vertical_lines, axis=0) > height * 0.5)[0]
            
            return f"""
            Table Dimensions: {width}x{height} pixels
            Estimated Rows: {len(row_positions) - 1}
            Estimated Columns: {len(col_positions) - 1}
            Grid Structure: {'Regular grid detected' if len(row_positions) > 1 and len(col_positions) > 1 else 'Irregular structure'}
            Border Style: {'Bordered' if np.sum(horizontal_lines) > 0 and np.sum(vertical_lines) > 0 else 'Borderless'}
            """
        except Exception as e:
            logger.error(f"Error in structure description: {str(e)}")
            return "Unable to analyze table structure" 