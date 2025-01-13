from .document import router as document_router
from .query import router as query_router
from .stats import router as stats_router

__all__ = ['document_router', 'query_router', 'stats_router'] 