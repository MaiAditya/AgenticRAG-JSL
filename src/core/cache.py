from typing import Dict, Any
from loguru import logger
import json
from datetime import datetime
import os

class DocumentCache:
    def __init__(self, cache_dir: str = "cache/documents"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Initialized DocumentCache at {cache_dir}")

    def add_document(self, file_path: str, results: Dict[str, Any]) -> None:
        """Add processed document results to cache
        
        Args:
            file_path: Original document path
            results: Processing results to cache
        """
        try:
            # Create a cache entry with timestamp
            cache_entry = {
                "file_path": file_path,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }
            
            # Generate cache file path
            file_name = os.path.basename(file_path)
            cache_file = os.path.join(
                self.cache_dir,
                f"{file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            # Write to cache file
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_entry, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Cached document results to {cache_file}")
            
        except Exception as e:
            logger.error(f"Error caching document {file_path}: {str(e)}")
            raise

    def get_document(self, file_path: str) -> Dict[str, Any] | None:
        """Retrieve cached results for a document
        
        Args:
            file_path: Original document path
            
        Returns:
            Cached results if found, None otherwise
        """
        try:
            file_name = os.path.basename(file_path)
            # List all cache files for this document
            cache_files = [
                f for f in os.listdir(self.cache_dir)
                if f.startswith(file_name) and f.endswith('.json')
            ]
            
            if not cache_files:
                return None
                
            # Get most recent cache file
            latest_cache = sorted(cache_files)[-1]
            cache_path = os.path.join(self.cache_dir, latest_cache)
            
            # Read and return cached results
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_entry = json.load(f)
            
            return cache_entry["results"]
            
        except Exception as e:
            logger.error(f"Error retrieving cache for {file_path}: {str(e)}")
            return None 