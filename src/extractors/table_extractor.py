from transformers import DetrImageProcessor, TableTransformerForObjectDetection
from PIL import Image
import torch
import numpy as np
from loguru import logger
import os
import cv2
from typing import List, Dict, Any, Tuple
import io
import json
import pytesseract
import re
import base64
from openai import OpenAI
import time

class TableExtractor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing TableExtractor on {self.device}")
        
        # Initialize Table Transformer for detection
        self.detector_processor = DetrImageProcessor.from_pretrained(
            "microsoft/table-transformer-detection",
            do_resize=True,
            size={'shortest_edge': 800, 'longest_edge': 1333},
        )
        self.detector = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection"
        ).to(self.device)
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Create output directories for debugging
        os.makedirs("logs/table_detections/originals", exist_ok=True)
        os.makedirs("logs/table_detections/visualizations", exist_ok=True)
        os.makedirs("logs/table_detections/cells", exist_ok=True)

    async def extract(self, image: Any) -> Dict[str, Any]:
        """Extract tables, generate descriptions, and metadata"""
        try:
            logger.info("Starting table extraction process")
            
            # Create log directories
            json_log_dir = "logs/table_detections/json"
            os.makedirs(json_log_dir, exist_ok=True)
            
            # Image conversion logging
            logger.debug(f"Input image type: {type(image)}")
            
            # Process image and get PIL format
            pil_image = await self._process_input_image(image)
            
            # Save for debugging with detailed logging
            timestamp = int(time.time())
            debug_path = os.path.join("logs/table_detections/originals", f"table_{timestamp}.png")
            pil_image.save(debug_path)
            logger.debug(f"Saved original image to: {debug_path}")
            
            # Table detection with enhanced logging
            logger.info("Running table detection model")
            detection_result = await self._detect_tables(pil_image)
            logger.info(f"Table detection completed. Found {len(detection_result['tables'])} tables")
            
            tables = []
            for idx, table_data in enumerate(detection_result['tables']):
                logger.info(f"Processing table {idx+1}/{len(detection_result['tables'])}")
                
                # Generate description with detailed logging
                logger.info(f"Generating vision description for table {idx+1}")
                description = await self._generate_vision_description(table_data['image'])
                logger.info("Vision Description generated successfully")
                logger.debug(f"Description length: {len(description)} characters")
                
                # Extract metadata with enhanced logging
                logger.info(f"Extracting metadata for table {idx+1}")
                metadata = await self._extract_table_metadata(table_data['image'])
                logger.info("Metadata extraction completed")
                logger.debug(f"Metadata keys: {list(metadata.keys())}")
                
                # Create comprehensive table record
                table_record = {
                    "table_id": f"table_{timestamp}_{idx}",
                    "bbox": table_data['bbox'],
                    "confidence": table_data['confidence'],
                    "description": description,
                    "metadata": metadata,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "vector_ready": True  # Flag for vector storage
                }
                
                # Log complete table analysis
                logger.info(f"Table {idx+1} Analysis Summary:")
                logger.info(json.dumps({
                    "table_id": table_record["table_id"],
                    "bbox_size": [b[1] - b[0] for b in zip(table_record["bbox"][:2], table_record["bbox"][2:])],
                    "confidence": table_record["confidence"],
                    "description_length": len(description),
                    "metadata_keys": list(metadata.keys())
                }, indent=2))
                
                # Save detailed JSON log
                json_path = os.path.join(json_log_dir, f"table_{timestamp}_{idx}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(table_record, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved detailed table analysis to: {json_path}")
                
                tables.append(table_record)
            
            logger.info(f"Successfully processed {len(tables)} tables")
            return {"tables": tables}

        except Exception as e:
            logger.error(f"Error in table extraction: {str(e)}", exc_info=True)
            return {"error": str(e)}

    async def _generate_vision_description(self, table_image: Image.Image) -> str:
        """Generate detailed description using OpenAI's GPT-4 Vision"""
        try:
            logger.info("Starting vision description generation")
            
            # Convert image to base64
            buffered = io.BytesIO()
            table_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Log the prompt being sent
            prompt = """As a medical professional, analyze this table in detail. Provide:
            1. Main medical topic/condition
            2. Key data points and clinical significance
            3. Table organization and structure
            4. Critical medical implications or warnings
            5. Relevance for healthcare providers
            Be specific and thorough in your medical analysis."""
            
            logger.info("Sending prompt to vision model:")
            logger.info(prompt)
            
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
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            description = response.choices[0].message.content
            
            # Log the full description with proper formatting
            logger.info("=== Vision Analysis Results ===")
            logger.info("Raw Description:")
            for line in description.split('\n'):
                logger.info(line.strip())
            
            # Parse and log structured sections
            sections = {
                "Medical Topic": "",
                "Key Data Points": "",
                "Table Structure": "",
                "Medical Implications": "",
                "Healthcare Relevance": ""
            }
            
            current_section = None
            for line in description.split('\n'):
                line = line.strip()
                if "topic" in line.lower() or "condition" in line.lower():
                    current_section = "Medical Topic"
                elif "data point" in line.lower() or "significance" in line.lower():
                    current_section = "Key Data Points"
                elif "structure" in line.lower() or "organization" in line.lower():
                    current_section = "Table Structure"
                elif "warning" in line.lower() or "implication" in line.lower():
                    current_section = "Medical Implications"
                elif "relevance" in line.lower() or "provider" in line.lower():
                    current_section = "Healthcare Relevance"
                
                if current_section and line:
                    sections[current_section] += line + "\n"
            
            # Log structured sections
            logger.info("\n=== Structured Analysis ===")
            for section, content in sections.items():
                if content.strip():
                    logger.info(f"\n{section}:")
                    logger.info(content.strip())
            
            return description

        except Exception as e:
            logger.error(f"Error generating vision description: {str(e)}", exc_info=True)
            return "Error generating description"

    async def _extract_table_metadata(self, table_image: Image.Image) -> Dict[str, Any]:
        """Enhanced metadata extraction with improved cell detection and OCR"""
        try:
            logger.debug("Starting metadata extraction")
            
            # Convert and preprocess
            table_np = np.array(table_image)
            gray = cv2.cvtColor(table_np, cv2.COLOR_RGB2GRAY)
            logger.debug(f"Image converted to grayscale: {gray.shape}")
            
            # Get structure description
            logger.debug("Analyzing table structure")
            structure_desc = self._get_structure_description(table_image)
            logger.debug(f"Structure analysis complete: {structure_desc}")
            
            # Get content summary
            logger.debug("Analyzing content patterns")
            content_summary = self._summarize_table_content(table_image)
            logger.debug(f"Content analysis complete: {content_summary}")
            
            # Combine metadata
            metadata = {
                "structure": structure_desc,
                "content": content_summary,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info("Metadata extraction completed successfully")
            return metadata

        except Exception as e:
            logger.error(f"Error in metadata extraction: {str(e)}", exc_info=True)
            return None

    def _detect_cells(self, grid: np.ndarray, table_np: np.ndarray) -> List[Dict[str, Any]]:
        """Detect and extract cells from table grid"""
        try:
            # Find contours in the grid
            contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and sort contours
            min_area = (grid.shape[0] * grid.shape[1]) / (100 * 100)
            cell_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            cells = []
            for contour in cell_contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract cell image
                cell_img = table_np[y:y+h, x:x+w]
                
                # Perform OCR on cell
                cell_text = pytesseract.image_to_string(
                    Image.fromarray(cell_img),
                    config='--psm 6'
                ).strip()
                
                # Determine if cell is header (typically in first row)
                is_header = y < table_np.shape[0] * 0.15
                
                cells.append({
                    "bbox": [x, y, x+w, y+h],
                    "text": cell_text,
                    "is_header": is_header,
                    "area": w * h
                })
            
            return cells

        except Exception as e:
            logger.error(f"Error in cell detection: {str(e)}")
            return []

    def _analyze_table_structure(self, cells: List[Dict], table_size: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze table structure including rows, columns, and headers"""
        try:
            if not cells:
                return {
                    "num_rows": 0,
                    "num_cols": 0,
                    "headers": [],
                    "layout": "unknown"
                }

            # Sort cells by position
            sorted_by_y = sorted(cells, key=lambda c: c["bbox"][1])
            
            # Detect rows by clustering y-coordinates
            row_clusters = self._cluster_coordinates([c["bbox"][1] for c in cells])
            num_rows = len(row_clusters)
            
            # Detect columns by clustering x-coordinates
            col_clusters = self._cluster_coordinates([c["bbox"][0] for c in cells])
            num_cols = len(col_clusters)
            
            # Extract headers
            headers = [c["text"] for c in cells if c["is_header"]]
            
            # Determine layout type
            layout = self._determine_layout_type(cells, row_clusters, col_clusters)
            
            return {
                "num_rows": num_rows,
                "num_cols": num_cols,
                "headers": headers,
                "layout": layout,
                "row_positions": row_clusters,
                "col_positions": col_clusters
            }

        except Exception as e:
            logger.error(f"Error in structure analysis: {str(e)}")
            return None

    def _analyze_data_types(self, cells: List[Dict]) -> Dict[str, List[str]]:
        """Analyze and classify data types in table cells"""
        try:
            data_types = {"numeric": [], "text": [], "date": [], "mixed": []}
            
            for cell in cells:
                text = cell["text"]
                if not text:
                    continue
                    
                # Check for numeric values
                if text.replace(".", "").replace("-", "").isdigit():
                    data_types["numeric"].append(text)
                
                # Check for dates using regex
                elif re.match(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', text):
                    data_types["date"].append(text)
                
                # Check for mixed content
                elif any(char.isdigit() for char in text) and any(char.isalpha() for char in text):
                    data_types["mixed"].append(text)
                
                # Default to text
                else:
                    data_types["text"].append(text)
            
            return data_types

        except Exception as e:
            logger.error(f"Error in data type analysis: {str(e)}")
            return {}

    def _determine_table_type(self, cells: List[Dict], structure: Dict) -> str:
        """Determine the type and purpose of the table"""
        try:
            headers = structure["headers"]
            data_types = self._analyze_data_types(cells)
            
            # Check for common medical table patterns
            if any("dosage" in h.lower() for h in headers):
                return "medication_dosage"
            elif any("diagnosis" in h.lower() for h in headers):
                return "diagnostic"
            elif any("treatment" in h.lower() for h in headers):
                return "treatment_protocol"
            elif len(data_types["numeric"]) > len(cells) * 0.7:
                return "numerical_data"
            elif len(data_types["date"]) > len(cells) * 0.3:
                return "temporal_data"
            else:
                return "general_medical"

        except Exception as e:
            logger.error(f"Error determining table type: {str(e)}")
            return "unknown"

    def _cluster_coordinates(self, coordinates: List[float], threshold: float = 10) -> List[float]:
        """Helper function to cluster coordinates for row/column detection"""
        if not coordinates:
            return []
        
        coordinates = sorted(coordinates)
        clusters = [[coordinates[0]]]
        
        for coord in coordinates[1:]:
            if coord - clusters[-1][-1] > threshold:
                clusters.append([])
            clusters[-1].append(coord)
        
        return [sum(cluster)/len(cluster) for cluster in clusters]

    def _determine_layout_type(self, cells: List[Dict], rows: List[float], cols: List[float]) -> str:
        """Determine the layout type of the table"""
        try:
            # Check for regular grid
            cell_count = len(cells)
            expected_cells = len(rows) * len(cols)
            
            if abs(cell_count - expected_cells) <= 2:
                return "regular_grid"
            elif len(rows) == 1:
                return "single_row"
            elif len(cols) == 1:
                return "single_column"
            else:
                return "irregular_grid"

        except Exception as e:
            logger.error(f"Error determining layout type: {str(e)}")
            return "unknown"

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
        """Generate a detailed description of the table using both BLIP-2 and LLaMA-2"""
        try:
            if not isinstance(table_image, Image.Image):
                logger.error(f"Invalid image type: {type(table_image)}")
                return "Error: Invalid image format"

            # Step 1: Get visual description using BLIP-2
            inputs = self.blip_processor(images=table_image, return_tensors="pt").to(self.device)
            visual_outputs = self.blip_model.generate(
                **inputs,
                max_length=150,  # Increased for better descriptions
                num_beams=5,     # Added beam search
                min_length=50,   # Ensure minimum length
                top_p=0.9,
                repetition_penalty=1.5
            )
            visual_description = self.blip_processor.decode(visual_outputs[0], skip_special_tokens=True)
            
            # Step 2: Get OCR text and structure
            ocr_text = pytesseract.image_to_string(table_image)
            structure_desc = self._get_structure_description(table_image)
            
            # Step 3: Use LLaMA to combine visual and textual information
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                llm_int8_enable_fp32_cpu_offload=True
            )
            
            model_name = "meta-llama/Llama-2-7b-chat-hf"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=quantization_config
            )

            # Format prompt in Llama-2 chat style
            chat_prompt = f"""<s>[INST] You are a medical professional analyzing a table from a medical document. 
            
            Here is the information about the table:

            Visual Analysis:
            {visual_description}

            OCR Text Content:
            {ocr_text[:500]}

            Table Structure:
            {structure_desc}

            Please provide a detailed medical analysis of this table covering:
            1. The main medical topic or condition being presented
            2. Key clinical information and data points
            3. How the information is organized
            4. Important medical implications or warnings
            5. Relevance for healthcare providers

            Focus on medical accuracy and clinical relevance. [/INST]

            Based on the provided information, here is my analysis of the medical table:"""

            # Generate with better parameters
            inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_length=1000,
                    min_length=200,
                    temperature=0.7,
                    top_p=0.9,
                    num_beams=3,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            description = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the response part after the prompt
            description = description.split("[/INST]")[-1].strip()
            
            # Clean up memory
            del model
            del tokenizer
            torch.cuda.empty_cache()
            
            return description

        except Exception as e:
            logger.error(f"Error generating table description: {str(e)}")
            return "Error generating description"

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