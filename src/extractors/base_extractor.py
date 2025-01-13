from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, content: Any) -> Dict[str, Any]:
        pass

    @abstractmethod
    def preprocess(self, content: Any) -> Any:
        pass 