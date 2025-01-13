from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseExtractor(ABC):
    @abstractmethod
    def preprocess(self, content: Any) -> Any:
        pass
    
    @abstractmethod
    async def extract(self, content: Any) -> Dict[str, Any]:
        pass 