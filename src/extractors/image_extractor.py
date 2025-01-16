from PIL import Image
from .base_extractor import BaseExtractor
import torch
import numpy as np
from loguru import logger
import io
import os
import datetime
import cv2
import json
import pytesseract
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms
from typing import List, Dict, Any

class ImageExtractor(BaseExtractor):
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageExtractor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        super().__init__()
        try:
            # Test pytesseract installation
            try:
                pytesseract.get_tesseract_version()
            except EnvironmentError:
                logger.error("Tesseract is not installed. Please install tesseract-ocr first.")
                raise
            except ImportError:
                logger.error("Pytesseract is not installed. Please install python-tesseract.")
                raise
            
            # Initialize MobileNetV3 (much lighter than VLM)
            self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model.eval()
            
            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
            
            # Create output directories
            os.makedirs("logs/image_extractions/originals", exist_ok=True)
            os.makedirs("logs/image_extractions/visualizations", exist_ok=True)
            os.makedirs("logs/image_extractions/preprocessed", exist_ok=True)
            os.makedirs("logs/image_extractions/ocr_results", exist_ok=True)
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing ImageExtractor: {str(e)}")
            raise

    def _analyze_image_structure(self, image: Image.Image) -> dict:
        """Analyze image structure using OpenCV"""
        # Convert PIL to CV2
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Edge detection for structural analysis
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
        
        # Detect shapes
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze structure
        num_horizontal_lines = 0
        num_vertical_lines = 0
        shapes = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                if angle < 30 or angle > 150:
                    num_vertical_lines += 1
                elif 60 < angle < 120:
                    num_horizontal_lines += 1
        
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
            shapes.append(len(approx))
        
        return {
            "num_horizontal_lines": num_horizontal_lines,
            "num_vertical_lines": num_vertical_lines,
            "num_shapes": len(contours),
            "shape_types": shapes
        }

    def _classify_image_type(self, features: dict) -> str:
        """Classify image type based on structural features"""
        h_lines = features["num_horizontal_lines"]
        v_lines = features["num_vertical_lines"]
        shapes = features["num_shapes"]
        
        if h_lines > 5 and v_lines > 5:
            return "flowchart" if shapes > 10 else "diagram"
        elif h_lines > 10 or v_lines > 10:
            return "diagram"
        else:
            return "regular"

    async def extract(self, image) -> dict:
        try:
            # Get metadata if available
            metadata = image.get('metadata', {}) if isinstance(image, dict) else {}
            
            # Preprocess image
            image = self.preprocess(image)
            
            # Save original image
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            original_path = f"logs/image_extractions/originals/original_{timestamp}.png"
            image.save(original_path)
            
            # Get basic image info
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

            # Analyze image structure
            structure_features = self._analyze_image_structure(image)
            image_type = self._classify_image_type(structure_features)
            
            # Enhanced OCR preprocessing for better text extraction
            try:
                # Convert to numpy array for OpenCV processing
                img_np = np.array(image)
                original = img_np.copy()
                
                # Convert to grayscale
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                
                # Multi-stage preprocessing pipeline
                # 1. Adaptive thresholding for better text separation
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 21, 10
                )
                
                # 2. Noise removal
                denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
                
                # 3. CLAHE enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(denoised)
                
                # 4. Morphological operations for text clarity
                kernel = np.ones((2,2), np.uint8)
                morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
                
                # 5. Contrast enhancement
                normalized = cv2.normalize(morph, None, 0, 255, cv2.NORM_MINMAX)
                
                # 6. Scale up with better interpolation
                scale_factor = 3
                scaled = cv2.resize(normalized, None, 
                                  fx=scale_factor, 
                                  fy=scale_factor, 
                                  interpolation=cv2.INTER_CUBIC)
                
                # 7. Border addition
                bordered = cv2.copyMakeBorder(
                    scaled, 20, 20, 20, 20,
                    cv2.BORDER_CONSTANT,
                    value=[255,255,255]
                )
                
                # Multi-pass OCR with different configurations
                ocr_results = []
                
                # Configuration variations for different text types
                configs = [
                    # Standard text
                    r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()/.+-%[],:;\'\" " -c tessedit_char_blacklist="|~`@#$^&*_={};<>?"',
                    # Dense text
                    r'--oem 3 --psm 4',
                    # Sparse text
                    r'--oem 3 --psm 11',
                    # Single column
                    r'--oem 3 --psm 3'
                ]
                
                for config in configs:
                    # Get detailed OCR data
                    ocr_data = pytesseract.image_to_data(
                        bordered,
                        config=config,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Process each word with confidence
                    for i in range(len(ocr_data['text'])):
                        conf = float(ocr_data['conf'][i])
                        text = ocr_data['text'][i].strip()
                        
                        if conf > 60 and text and len(text) > 1:
                            ocr_results.append({
                                'text': text,
                                'confidence': conf,
                                'bbox': [
                                    ocr_data['left'][i],
                                    ocr_data['top'][i],
                                    ocr_data['width'][i],
                                    ocr_data['height'][i]
                                ],
                                'page_num': ocr_data['page_num'][i],
                                'block_num': ocr_data['block_num'][i],
                                'par_num': ocr_data['par_num'][i],
                                'line_num': ocr_data['line_num'][i],
                                'word_num': ocr_data['word_num'][i]
                            })
                
                # Post-process OCR results
                processed_results = self._post_process_ocr(ocr_results)
                
                # Save debug information
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_path = f"logs/image_extractions/preprocessed/ocr_{timestamp}.png"
                cv2.imwrite(debug_path, bordered)
                
                # Save detailed OCR results
                ocr_debug = {
                    "timestamp": timestamp,
                    "image_path": debug_path,
                    "raw_results": ocr_results,
                    "processed_results": processed_results,
                    "preprocessing_steps": [
                        "Adaptive thresholding",
                        "Noise removal",
                        "CLAHE enhancement",
                        "Morphological operations",
                        "Contrast normalization",
                        "Scale up with cubic interpolation",
                        "Border addition"
                    ]
                }
                
                debug_json_path = f"logs/image_extractions/ocr_results/ocr_{timestamp}.json"
                os.makedirs(os.path.dirname(debug_json_path), exist_ok=True)
                with open(debug_json_path, 'w') as f:
                    json.dump(ocr_debug, f, indent=2)
                
                return {
                    "type": "image",
                    "image_type": image_type,
                    "image_details": image_info,
                    "structure_features": structure_features,
                    "text_content": processed_results['text'],
                    "ocr_metadata": {
                        "confidence": processed_results['average_confidence'],
                        "word_count": len(processed_results['words']),
                        "layout": processed_results['layout'],
                        "debug_paths": {
                            "preprocessed_image": debug_path,
                            "ocr_results": debug_json_path
                        }
                    },
                    "metadata": metadata
                }
                
            except ImportError:
                logger.error("Pytesseract not installed. OCR functionality is required.")
                raise
            except Exception as e:
                logger.error(f"Error in OCR processing: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error in image extraction: {str(e)}")
            return {"error": str(e)}

    def _post_process_ocr(self, ocr_results: List[Dict]) -> Dict:
        """Post-process OCR results to improve quality and structure"""
        # Sort results by position (top to bottom, left to right)
        sorted_results = sorted(
            ocr_results,
            key=lambda x: (x['block_num'], x['par_num'], x['line_num'], x['word_num'])
        )
        
        # Group words by lines
        lines = {}
        for result in sorted_results:
            line_key = f"{result['block_num']}_{result['par_num']}_{result['line_num']}"
            if line_key not in lines:
                lines[line_key] = []
            lines[line_key].append(result)
        
        # Combine words into coherent text
        processed_text = []
        for line in lines.values():
            line_text = ' '.join(word['text'] for word in line)
            processed_text.append(line_text)
        
        return {
            'text': '\n'.join(processed_text),
            'words': [r['text'] for r in sorted_results],
            'average_confidence': sum(r['confidence'] for r in sorted_results) / len(sorted_results),
            'layout': {
                'blocks': len(set(r['block_num'] for r in sorted_results)),
                'paragraphs': len(set(f"{r['block_num']}_{r['par_num']}" for r in sorted_results)),
                'lines': len(lines)
            }
        }

    def preprocess(self, image: Any) -> Image.Image:
        """Preprocess the image for analysis
        
        Args:
            image: Input image (can be PIL Image, bytes, or dict with image data)
            
        Returns:
            PIL Image ready for processing
        """
        try:
            if isinstance(image, dict) and 'image' in image:
                image = image['image']
            
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            elif hasattr(image, 'tobytes'):
                img_data = image.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
            elif not isinstance(image, Image.Image):
                raise ValueError("Unsupported image format")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            raise