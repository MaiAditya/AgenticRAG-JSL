from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoProcessor, AutoModelForVision2Seq
from paddleocr import PaddleOCR
from PIL import Image
from .base_extractor import BaseExtractor
from src.core.config import settings
import torch
import numpy as np
from loguru import logger
import io
import json
import os
import datetime
import cv2
from openai import OpenAI
import base64
from typing import Dict, Any
import time
import fitz

class ImageExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        try:
            logger.info("Initializing ImageExtractor...")
            
            # Initialize OpenAI client for vision analysis
            self.client = OpenAI()
            
            # Initialize BLIP-2 model for backup/supplementary analysis
            logger.info("Loading BLIP-2 model...")
            self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            logger.info("BLIP-2 model loaded successfully")
            
            # Initialize PaddleOCR
            logger.info("Initializing PaddleOCR...")
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
            logger.info("PaddleOCR initialized successfully")
            
            # Move models to device
            self.device = torch.device(settings.DEVICE)
            self.blip_model.to(self.device)
            logger.info(f"Models moved to device: {self.device}")
            
            # Create output directories
            logger.info("Creating output directories...")
            os.makedirs("logs/image_extractions/originals", exist_ok=True)
            os.makedirs("logs/image_extractions/visualizations", exist_ok=True)
            os.makedirs("logs/image_extractions/elements", exist_ok=True)
            logger.info("Output directories created successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ImageExtractor: {str(e)}")
            raise

    async def preprocess(self, image) -> Image.Image:
        try:
            logger.info("Starting image preprocessing")
            
            if isinstance(image, Image.Image):
                return image
            
            if isinstance(image, fitz.Pixmap):
                # Convert Pixmap to PIL Image
                img_data = image.samples
                if image.n >= 4:  # RGBA or CMYK
                    # Convert to RGB
                    pix = fitz.Pixmap(fitz.csRGB, image)
                    img_data = pix.samples
                    pix = None
                
                return Image.frombytes(
                    "RGB", 
                    [image.width, image.height], 
                    img_data
                )
            
            # Handle bytes or BytesIO input
            if isinstance(image, (bytes, io.BytesIO)):
                if isinstance(image, bytes):
                    image = io.BytesIO(image)
                return Image.open(image).convert('RGB')
            
            raise ValueError(f"Unsupported image type: {type(image)}")
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            raise

    def _detect_visual_type(self, image: Image.Image) -> str:
        """Detect the type of visual element (flowchart, diagram, regular image, etc.)"""
        # Convert to numpy array for OpenCV processing
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Detect lines and shapes
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        # Count connected components
        _, labels = cv2.connectedComponents(edges)
        
        if lines is not None and len(lines) > 20:
            return "flowchart" if len(lines) > 50 else "diagram"
        return "image"

    async def extract(self, image) -> dict:
        try:
            logger.info("Starting image extraction process...")
            
            # Process image
            processed_image = await self.preprocess(image)
            
            # Detect visual type
            visual_type = self._detect_visual_type(processed_image)
            logger.info(f"Detected visual type: {visual_type}")
            
            # Get GPT-4 Vision analysis
            vision_analysis = await self._generate_vision_analysis(processed_image, visual_type)
            
            # Extract OCR text
            ocr_result = self.ocr.ocr(np.array(processed_image))
            extracted_text = [line[1][0] for line in ocr_result[0]] if ocr_result and ocr_result[0] else []
            
            # Save debug image
            timestamp = int(time.time())
            debug_path = f"logs/image_extractions/originals/image_{timestamp}.jpg"  # Use JPEG instead of PNG
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            processed_image.convert('RGB').save(debug_path, 'JPEG')  # Convert to RGB and save as JPEG
            
            result = {
                "type": "visual_element",
                "visual_type": visual_type,
                "description": vision_analysis.get("raw_analysis", ""),
                "structured_analysis": vision_analysis.get("structured_analysis", {}),
                "metadata": {
                    "extracted_text": extracted_text,
                    "element_count": len(extracted_text),
                    "dimensions": processed_image.size,
                    "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "text_content": " ".join(extracted_text)
                },
                "text_content": " ".join(extracted_text),
                "vector_ready": True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in image extraction: {str(e)}")
            return {
                "error": str(e),
                "type": "error",
                "text_content": "",
                "metadata": {}
            }

    async def _extract_visual_info(self, image: Image.Image, visual_type: str) -> dict:
        """Extract detailed information based on visual type"""
        try:
            # Get detailed caption from BLIP-2
            inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.blip_model.generate(**inputs, max_length=100)
            caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract text using PaddleOCR
            ocr_result = self.ocr.ocr(np.array(image))
            extracted_text = [line[1][0] for line in ocr_result[0]] if ocr_result[0] else []
            
            # Combine information based on visual type
            if visual_type == "flowchart":
                elements = self._process_flowchart(image, ocr_result)
            elif visual_type == "diagram":
                elements = self._process_diagram(image, ocr_result)
            else:
                elements = []
            
            return {
                "description": caption,
                "metadata": {
                    "extracted_text": extracted_text,
                    "element_count": len(elements),
                    "text_density": len(extracted_text)
                },
                "elements": elements,
                "extracted_text": extracted_text
            }
        except Exception as e:
            logger.error(f"Error in visual info extraction: {str(e)}")
            return {
                "description": "Error extracting visual information",
                "metadata": {},
                "elements": [],
                "extracted_text": []
            }

    def _process_flowchart(self, image: Image.Image, ocr_result) -> list:
        """Process flowchart-specific elements"""
        elements = []
        img_np = np.array(image)
        
        # Detect shapes
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            shape_type = self._classify_shape(contour)
            
            # Find text within shape
            shape_text = ""
            for text_line in ocr_result[0] if ocr_result[0] else []:
                text_box = text_line[0]
                if self._is_point_inside((text_box[0][0], text_box[0][1]), (x, y, w, h)):
                    shape_text = text_line[1][0]
                    break
            
            elements.append({
                "type": "flowchart_element",
                "shape": shape_type,
                "text": shape_text,
                "position": {"x": x, "y": y, "width": w, "height": h}
            })
        
        return elements

    def _process_diagram(self, image: Image.Image, ocr_result) -> list:
        """Process diagram-specific elements"""
        elements = []
        img_np = np.array(image)
        
        # Detect regions of interest
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            if area > 100:  # Filter out noise
                # Extract text in this region
                region_text = []
                for text_line in ocr_result[0] if ocr_result[0] else []:
                    text_box = text_line[0]
                    if self._is_point_inside((text_box[0][0], text_box[0][1]), (x, y, w, h)):
                        region_text.append(text_line[1][0])
                
                elements.append({
                    "type": "diagram_element",
                    "text": " ".join(region_text),
                    "position": {"x": x, "y": y, "width": w, "height": h},
                    "area": area
                })
        
        return elements

    def _classify_shape(self, contour) -> str:
        """Classify the type of shape based on contour properties"""
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        num_vertices = len(approx)
        
        if num_vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            if 0.95 <= aspect_ratio <= 1.05:
                return "rectangle"
            else:
                return "rectangle"
        elif num_vertices == 3:
            return "triangle"
        elif num_vertices > 4:
            return "ellipse"
        else:
            return "unknown"

    def _create_visualization(self, image: Image.Image, visual_info: dict) -> np.ndarray:
        """Create a visualization of detected elements"""
        img_np = np.array(image)
        vis_image = img_np.copy()
        
        # Draw bounding boxes for detected elements
        for element in visual_info.get("elements", []):
            if "bbox" in element:
                x, y, w, h = element["bbox"]
                color = {
                    "rectangle": (0, 255, 0),
                    "triangle": (0, 0, 255),
                    "ellipse": (255, 0, 0),
                    "unknown": (128, 128, 128)
                }.get(element.get("shape", "unknown"), (128, 128, 128))
                
                cv2.rectangle(vis_image, (int(x), int(y)), 
                             (int(x + w), int(y + h)), color, 2)
                
                # Add text label if available
                if "text" in element and element["text"]:
                    cv2.putText(vis_image, element["text"][:20], 
                               (int(x), int(y - 5)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_image

    def _is_point_inside(self, point: tuple, bbox: tuple) -> bool:
        """Check if a point is inside a bounding box
        
        Args:
            point (tuple): (x, y) coordinates of the point
            bbox (tuple): (x, y, width, height) of the bounding box
        
        Returns:
            bool: True if point is inside bbox, False otherwise
        """
        px, py = point
        x, y, w, h = bbox
        return (x <= px <= x + w) and (y <= py <= y + h)

    async def _generate_vision_analysis(self, image: Image.Image, visual_type: str) -> Dict[str, Any]:
        """Generate detailed analysis using GPT-4 Vision based on visual type"""
        try:
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Create type-specific prompt
            prompts = {
                "flowchart": """Analyze this medical flowchart in detail. Provide:
                    1. Main clinical process/pathway being illustrated
                    2. Key decision points and their criteria
                    3. Important clinical steps and their sequence
                    4. Critical medical considerations or warnings
                    5. Relevance for healthcare workflow
                    Be specific and thorough in your medical analysis.""",
                
                "diagram": """Analyze this medical diagram in detail. Provide:
                    1. Main anatomical/clinical concept illustrated
                    2. Key components and their relationships
                    3. Important medical details shown
                    4. Clinical significance and applications
                    5. Relevant medical implications
                    Be specific and thorough in your medical analysis.""",
                
                "image": """Analyze this medical image in detail. Provide:
                    1. Type of medical imaging/visualization shown
                    2. Key anatomical/clinical features visible
                    3. Notable findings or patterns
                    4. Clinical significance
                    5. Relevant diagnostic implications
                    Be specific and thorough in your medical analysis."""
            }
            
            prompt = prompts.get(visual_type, prompts["image"])
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_str}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            analysis = response.choices[0].message.content
            
            # Structure the analysis
            sections = {
                "Main Topic": "",
                "Key Components": "",
                "Clinical Details": "",
                "Medical Implications": "",
                "Healthcare Relevance": ""
            }
            
            # Parse sections from the response
            current_section = None
            for line in analysis.split('\n'):
                line = line.strip()
                for section in sections.keys():
                    if section.lower() in line.lower():
                        current_section = section
                        break
                if current_section and line:
                    sections[current_section] += line + "\n"
            
            return {
                "raw_analysis": analysis,
                "structured_analysis": sections,
                "visual_type": visual_type
            }
            
        except Exception as e:
            logger.error(f"Error in vision analysis: {str(e)}")
            return {
                "error": str(e),
                "visual_type": visual_type
            }