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

class ImageExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        try:
            # Initialize BLIP-2 model for detailed image captioning
            self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Initialize CogVLM for detailed visual understanding
            self.cogvlm_processor = AutoProcessor.from_pretrained("THUDM/cogvlm-chat-hf")
            self.cogvlm_model = AutoModelForVision2Seq.from_pretrained("THUDM/cogvlm-chat-hf")
            
            # Initialize PaddleOCR for text extraction
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
            
            # Move models to device
            self.device = torch.device(settings.DEVICE)
            self.blip_model.to(self.device)
            self.cogvlm_model.to(self.device)
            
            # Create output directories
            os.makedirs("logs/image_extractions/originals", exist_ok=True)
            os.makedirs("logs/image_extractions/visualizations", exist_ok=True)
            os.makedirs("logs/image_extractions/elements", exist_ok=True)
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
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
            if isinstance(image, dict) and 'image' in image:
                image = image['image']
                metadata = image.get('metadata', {})
            else:
                metadata = {}
            
            if hasattr(image, 'tobytes'):
                img_data = image.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                
                # Save original image
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                original_path = f"logs/image_extractions/originals/original_{timestamp}.png"
                image.save(original_path)
                
                # Detect visual type
                visual_type = self._detect_visual_type(image)
                
                # Extract information based on visual type
                visual_info = await self._extract_visual_info(image, visual_type)
                
                # Create structured metadata
                metadata = {
                    "type": visual_type,
                    "timestamp": timestamp,
                    "size": image.size,
                    "format": image.format,
                    "mode": image.mode,
                    **visual_info.get("metadata", {})
                }
                
                # Generate visualization
                vis_image = self._create_visualization(image, visual_info)
                vis_path = f"logs/image_extractions/visualizations/vis_{timestamp}.png"
                cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                
                return {
                    "type": "visual_element",
                    "visual_type": visual_type,
                    "description": visual_info["description"],
                    "metadata": metadata,
                    "extracted_text": visual_info.get("extracted_text", []),
                    "elements": visual_info.get("elements", []),
                    "debug_paths": {
                        "original": original_path,
                        "visualization": vis_path
                    }
                }
                
        except Exception as e:
            logger.error(f"Error in image extraction: {str(e)}")
            return {"error": str(e)}

    async def _extract_visual_info(self, image: Image.Image, visual_type: str) -> dict:
        """Extract detailed information based on visual type"""
        # Get detailed caption from BLIP-2
        inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.blip_model.generate(**inputs, max_length=100)
        caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
        
        # Get detailed analysis from CogVLM
        cogvlm_inputs = self.cogvlm_processor(images=image, return_tensors="pt").to(self.device)
        cogvlm_outputs = self.cogvlm_model.generate(**cogvlm_inputs, max_length=200)
        detailed_analysis = self.cogvlm_processor.decode(cogvlm_outputs[0], skip_special_tokens=True)
        
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
            "description": f"{caption}\n\nDetailed Analysis: {detailed_analysis}",
            "metadata": {
                "extracted_text": extracted_text,
                "element_count": len(elements),
                "text_density": len(extracted_text)
            },
            "elements": elements,
            "extracted_text": extracted_text
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