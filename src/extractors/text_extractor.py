from unstructured.partition.auto import partition
from transformers import AutoTokenizer, AutoModel
from .base_extractor import BaseExtractor
from typing import List, Dict, Any

class TextExtractor(BaseExtractor):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    def preprocess(self, content: str) -> List[str]:
        elements = partition(content)
        return [str(element) for element in elements]
    
    def extract(self, content: str) -> Dict[str, Any]:
        processed_text = self.preprocess(content)
        return {
            "type": "text",
            "content": processed_text,
            "metadata": {
                "source": "text_extractor",
                "chunks": len(processed_text)
            }
        } 