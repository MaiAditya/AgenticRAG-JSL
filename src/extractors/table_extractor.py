from transformers import DetrImageProcessor, TableTransformerForObjectDetection
import torch
from PIL import Image
from .base_extractor import BaseExtractor
from core.config import settings

class TableExtractor(BaseExtractor):
    def __init__(self):
        self.processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        self.model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        self.model.to(settings.DEVICE)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs.to(settings.DEVICE)

    def extract(self, image: Image.Image) -> dict:
        inputs = self.preprocess(image)
        outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=target_sizes
        )[0]

        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            tables.append({
                "score": score.item(),
                "label": self.model.config.id2label[label.item()],
                "box": box.tolist()
            })

        return {
            "type": "table",
            "content": tables,
            "metadata": {
                "source": "table_extractor",
                "model": "table-transformer-detection"
            }
        } 