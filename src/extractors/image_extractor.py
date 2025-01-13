from transformers import ViTImageProcessor, ViTForImageClassification
from ultralytics import YOLO
from PIL import Image
from .base_extractor import BaseExtractor
from core.config import settings
import torch

class ImageExtractor(BaseExtractor):
    def __init__(self):
        # Initialize YOLO for object detection
        self.detector = YOLO('yolov8n.pt')
        
        # Initialize ViT for image understanding
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.model.to(settings.DEVICE)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.processor(images=image, return_tensors="pt").pixel_values.to(settings.DEVICE)

    def extract(self, image: Image.Image) -> dict:
        # Detect objects in image
        detections = self.detector(image)[0]
        
        # Process whole image with ViT
        inputs = self.preprocess(image)
        outputs = self.model(inputs)
        
        # Extract detected objects and their classifications
        objects = []
        for det in detections.boxes.data:
            x1, y1, x2, y2, conf, cls = det
            obj_image = image.crop((x1, y1, x2, y2))
            obj_features = self.model(self.preprocess(obj_image))
            
            objects.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(conf),
                "class": detections.names[int(cls)],
                "features": obj_features.logits.tolist()
            })

        return {
            "type": "image",
            "content": {
                "objects": objects,
                "global_features": outputs.logits.tolist()
            },
            "metadata": {
                "source": "yolo_vit",
                "num_objects": len(objects)
            }
        } 