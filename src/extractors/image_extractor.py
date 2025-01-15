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
from typing import Dict, Any, Union

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
            
            # Create output directories for logging
            self.log_dirs = {
                'originals': "logs/image_extractions/originals",
                'visualizations': "logs/image_extractions/visualizations",
                'captions': "logs/image_extractions/captions"
            }
            
            for dir_path in self.log_dirs.values():
                os.makedirs(dir_path, exist_ok=True)
            
            logger.info("ImageExtractor initialized with logging directories")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def preprocess(self, image: Image.Image) -> dict:
        """
        Implement the abstract preprocess method from BaseExtractor
        """
        try:
            # Basic image preprocessings 
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

    async def extract(self, image_input: Union[Dict, Image.Image, bytes]) -> Dict[str, Any]:
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Handle different input types
            if isinstance(image_input, dict):
                image = image_input['image']
                metadata = image_input.get('metadata', {})
                page_number = metadata.get('page_number', 'unknown')
            elif isinstance(image_input, Image.Image):
                image = image_input
                metadata = {}
                page_number = 'unknown'
            elif isinstance(image_input, bytes):
                image = Image.open(io.BytesIO(image_input))
                metadata = {}
                page_number = 'unknown'
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")

            # Save original image
            original_path = os.path.join(
                self.log_dirs['originals'], 
                f"original_page{page_number}_{timestamp}.png"
            )
            image.save(original_path)
            logger.info(f"Saved original image to {original_path}")

            # Process the image
            logger.info("Processing image for caption generation")
            inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.caption_model.generate(**inputs)
            caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)

            # Create visualization with caption
            vis_image = np.array(image)
            if vis_image.ndim == 2:  # Convert grayscale to RGB
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
            
            # Add caption text to visualization
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_color = (0, 255, 0)  # Green color
            
            # Calculate text size and position
            text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]
            text_x = 10
            text_y = vis_image.shape[0] - 20  # 20 pixels from bottom
            
            # Add semi-transparent background for text
            overlay = vis_image.copy()
            cv2.rectangle(
                overlay,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                (0, 0, 0),
                -1
            )
            vis_image = cv2.addWeighted(overlay, 0.6, vis_image, 0.4, 0)
            
            # Add caption text
            cv2.putText(
                vis_image,
                caption,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                font_thickness
            )
            
            # Save visualization
            vis_path = os.path.join(
                self.log_dirs['visualizations'],
                f"visualization_page{page_number}_{timestamp}.png"
            )
            cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved visualization with caption to {vis_path}")
            
            # Save caption to text file
            caption_path = os.path.join(
                self.log_dirs['captions'],
                f"caption_page{page_number}_{timestamp}.txt"
            )
            with open(caption_path, 'w') as f:
                json.dump({
                    'page_number': page_number,
                    'timestamp': timestamp,
                    'caption': caption,
                    'metadata': metadata
                }, f, indent=2)
            logger.info(f"Saved caption data to {caption_path}")

            result = {
                "caption": caption,
                "metadata": {
                    **metadata,
                    "size": image.size,
                    "mode": image.mode,
                    "format": image.format,
                    "logs": {
                        "original_image": original_path,
                        "visualization": vis_path,
                        "caption_file": caption_path
                    }
                }
            }
            
            logger.info(f"Successfully generated caption: {caption}")
            return result
            
        except Exception as e:
            logger.error(f"Error in image extraction: {str(e)}")
            return {"error": str(e)}