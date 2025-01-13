from loguru import logger
from typing import Any, Callable
from functools import wraps

def handle_extraction_error(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise ExtractionError(f"Failed to extract content: {str(e)}")
    return wrapper

class ExtractionError(Exception):
    pass

class ProcessingError(Exception):
    pass

def setup_error_handling():
    logger.add(
        "logs/error.log",
        rotation="500 MB",
        retention="10 days",
        level="ERROR"
    ) 