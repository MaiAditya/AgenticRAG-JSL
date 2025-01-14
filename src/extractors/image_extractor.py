from transformers import BlipProcessor, BlipForConditionalGeneration
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
            # Initialize BLIP model for image captioning
            model_name = "Salesforce/blip-image-captioning-base"
            self.caption_processor = BlipProcessor.from_pretrained(model_name)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.caption_model.to(settings.DEVICE)
            self.device = torch.device(settings.DEVICE)
            
            # Create output directories for debugging
            os.makedirs("logs/image_extractions/originals", exist_ok=True)
            os.makedirs("logs/image_extractions/visualizations", exist_ok=True)
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def preprocess(self, image: Image.Image) -> dict:
        """
        Implement the abstract preprocess method from BaseExtractor
        """
        try:
            # Basic image preprocessing
            image_info = {
                "size": image.size,
                "mode": image.mode,
                "format": image.format,
                "aspect_ratio": round(image.size[0] / image.size[1], 2),
                "resolution": f"{image.size[0]}x{image.size[1]}",
                "color_space": image.mode
            }
            
            # Prepare image for model
            inputs = self.caption_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            return {
                "image_info": image_info,
                "model_inputs": inputs
            }
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    async def extract(self, image) -> dict:
        try:
            if isinstance(image, dict) and 'image' in image:
                image = image['image']
                metadata = image.get('metadata', {})
            else:
                metadata = {}
            
            if hasattr(image, 'tobytes'):
                logger.debug("Converting Pixmap to PIL Image")
                img_data = image.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                
                # Save original image
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                original_path = f"logs/image_extractions/originals/original_{timestamp}.png"
                image.save(original_path)
                logger.debug(f"Saved original image to {original_path}")
                
                # Enhanced image analysis
                image_info = {
                    "size": image.size,
                    "mode": image.mode,
                    "format": image.format,
                    "aspect_ratio": round(image.size[0] / image.size[1], 2),
                    "resolution": f"{image.size[0]}x{image.size[1]}",
                    "color_space": image.mode,
                    "page_number": metadata.get("page_number"),
                    "location": metadata.get("location")
                }
                
                # Generate enhanced caption
                inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.caption_model.generate(
                        **inputs,
                        max_length=100,
                        num_beams=5,
                        length_penalty=1.5,
                        num_return_sequences=1,
                        temperature=0.7
                    )
                caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
                
                # Create visualization with caption
                vis_image = np.array(image)
                # Add caption as text on the image
                cv2.putText(
                    vis_image,
                    f"Caption: {caption[:100]}",  # Truncate long captions
                    (10, 30),  # Position
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,  # Font scale
                    (0, 255, 0),  # Color (BGR)
                    2  # Thickness
                )
                
                # Save visualization
                vis_path = f"logs/image_extractions/visualizations/vis_{timestamp}.png"
                cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                
                # Log comprehensive information
                logger.bind(image_extraction=True).info(
                    f"Image Analysis:\n"
                    f"Original Image: {original_path}\n"
                    f"Visualization: {vis_path}\n"
                    f"Resolution: {image_info['resolution']}\n"
                    f"Aspect Ratio: {image_info['aspect_ratio']}\n"
                    f"Color Space: {image_info['color_space']}\n"
                    f"Generated Caption: {caption}"
                )
                
                return {
                    "type": "image",
                    "caption": caption,
                    "image_details": image_info,
                    "metadata": metadata,
                    "debug_paths": {
                        "original": original_path,
                        "visualization": vis_path
                    }
                }
            
        except Exception as e:
            logger.error(f"Error in image extraction: {str(e)}")
            return {"error": str(e)}