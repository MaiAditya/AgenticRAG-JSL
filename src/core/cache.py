from functools import lru_cache
from typing import Any, Dict, Optional
import hashlib
import json

class DocumentCache:
    def __init__(self, max_size: int = 1000):
        self.cache = lru_cache(maxsize=max_size)(self._cache_func)

    def _cache_func(self, key: str) -> Dict[str, Any]:
        pass

    def get_cache_key(self, content: Any) -> str:
        if isinstance(content, (str, bytes)):
            return hashlib.md5(str(content).encode()).hexdigest()
        return hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def get(self, content: Any) -> Optional[Dict[str, Any]]:
        key = self.get_cache_key(content)
        return self.cache(key)

    def set(self, content: Any, value: Dict[str, Any]):
        key = self.get_cache_key(content)
        self.cache.__wrapped__[key] = value 