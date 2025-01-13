from functools import lru_cache
from typing import Any, Dict, Optional
import hashlib
import json

class DocumentCache:
    def __init__(self, max_size: int = 1000):
        self._cache = {}
        self.max_size = max_size

    def get_cache_key(self, content: Any) -> str:
        if isinstance(content, (str, bytes)):
            return hashlib.md5(str(content).encode()).hexdigest()
        return hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(key)

    def set(self, key: str, value: Dict[str, Any]):
        if len(self._cache) >= self.max_size:
            # Remove oldest item if cache is full
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value 