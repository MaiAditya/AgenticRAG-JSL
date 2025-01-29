from transformers import AutoTokenizer, AutoModel
from .base_extractor import BaseExtractor
from typing import List, Dict, Any, Optional
from loguru import logger
import os

class TextExtractor(BaseExtractor):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    def preprocess(self, content: str) -> List[str]:
        """Preprocess text content by splitting into meaningful chunks"""
        if not content:
            return []
            
        # Split content into paragraphs
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        
        # For each paragraph, split into sentences if too long
        elements = []
        for para in paragraphs:
            if len(para) > 500:  # arbitrary length threshold
                # Simple sentence splitting
                sentences = [s.strip() + '.' for s in para.split('.') if s.strip()]
                elements.extend(sentences)
            else:
                elements.append(para)
                
        return elements
    
    async def extract(self, content: str) -> Dict[str, Any]:
        """Extract text content from the given input."""
        try:
            if not content or not isinstance(content, str):
                return {
                    "type": "text",
                    "content": "",
                    "error": "Invalid content provided"
                }
                
            # Process the text content
            elements = self.preprocess(content)
            
            return {
                "type": "text",
                "content": content.strip(),
                "elements": elements
            }
            
        except Exception as e:
            logger.error(f"Error in TextExtractor: {str(e)}")
            return {
                "type": "text",
                "content": "",
                "error": str(e)
            } 