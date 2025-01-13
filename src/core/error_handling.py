from loguru import logger
from typing import Any, Callable
from functools import wraps
from fastapi import FastAPI

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

def setup_error_handling(app: FastAPI):
    # Setup logging
    logger.add(
        "logs/error.log",
        rotation="500 MB",
        retention="10 days",
        level="ERROR"
    )
    
    # Add exception handlers
    @app.exception_handler(ExtractionError)
    async def extraction_error_handler(request, exc):
        return {"error": str(exc)}, 422
    
    @app.exception_handler(ProcessingError)
    async def processing_error_handler(request, exc):
        return {"error": str(exc)}, 500 