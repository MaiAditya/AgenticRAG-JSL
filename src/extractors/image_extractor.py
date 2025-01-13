from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
from .base_extractor import BaseExtractor
from core.config import settings
import torch

class ImageExtractor(BaseExtractor):
    def __init__(self):
        self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        self.model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        self.model.to(settings.DEVICE)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        return pixel_values.to(settings.DEVICE)

    def extract(self, image: Image.Image) -> dict:
        pixel_values = self.preprocess(image)
        
        outputs = self.model.generate(
            pixel_values,
            max_length=512,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=4,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True
        )

        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")

        return {
            "type": "image",
            "content": sequence,
            "metadata": {
                "source": "image_extractor",
                "model": "donut-base-finetuned-docvqa"
            }
        } 