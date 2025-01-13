from transformers import ViTImageProcessor, ViTForImageClassification
from ultralytics import YOLO
from PIL import Image
from .base_extractor import BaseExtractor
from src.core.config import settings
import torch
import numpy as np
from loguru import logger
from src.core.error_handling import ProcessingError
import io
import json

class ImageExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        try:
            # Initialize YOLO with a specific version and model
            self.yolo_model = YOLO('yolov8n.pt')
            self.device = torch.device(settings.DEVICE)
            
            # Initialize ViT for image understanding
            self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            self.vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            self.vit_model.to(settings.DEVICE)
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.processor(images=image, return_tensors="pt").pixel_values.to(settings.DEVICE)

    async def extract(self, image) -> dict:
        try:
            # Handle different input types
            if isinstance(image, dict) and 'image' in image:
                image = image['image']
                metadata = image.get('metadata', {})
            else:
                metadata = {}
            
            # Convert PyMuPDF Pixmap to PIL Image
            if hasattr(image, 'tobytes'):
                logger.debug("Converting Pixmap to PIL Image")
                img_data = image.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                
                # Save image metadata
                image_info = {
                    "size": image.size,
                    "mode": image.mode,
                    "format": image.format,
                    "page_number": metadata.get("page_number"),
                    "location": metadata.get("location")
                }
                logger.bind(image_extraction=True).info(
                    f"Processing image:\n{json.dumps(image_info, indent=2)}"
                )

            # Use existing YOLO model
            results = self.yolo_model(image)
            
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    detection = {
                        "class": r.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist(),
                        "metadata": image_info
                    }
                    detections.append(detection)
                    logger.bind(image_extraction=True).info(
                        f"Detected object:\n{json.dumps(detection, indent=2)}"
                    )
            
            return {
                "type": "image",
                "detections": detections,
                "metadata": image_info
            }
            
        except Exception as e:
            logger.error(f"Error in image extraction: {str(e)}")
            return {"error": str(e)}