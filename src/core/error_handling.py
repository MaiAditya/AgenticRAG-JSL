from loguru import logger
from typing import Any, Callable
from functools import wraps
from fastapi import FastAPI
import os

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
    """Custom exception for document processing errors"""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

def setup_error_handling(app: FastAPI):
    # Create logs directory if it doesn't exist
    os.makedirs("logs/images", exist_ok=True)
    os.makedirs("logs/tables", exist_ok=True)
    
    # Setup logging configurations
    logger.add(
        "logs/error.log",
        rotation="500 MB",
        retention="10 days",
        level="ERROR",
        backtrace=True,
        diagnose=True
    )
    
    logger.add(
        "logs/info.log",
        rotation="500 MB",
        retention="10 days",
        level="INFO"
    )
    
    logger.add(
        "logs/images/extractions.log",
        rotation="500 MB",
        retention="10 days",
        level="INFO",
        filter=lambda record: "image_extraction" in record["extra"]
    )
    
    logger.add(
        "logs/tables/extractions.log",
        rotation="500 MB",
        retention="10 days",
        level="INFO",
        filter=lambda record: "table_extraction" in record["extra"]
    )
    
    # Add exception handlers
    @app.exception_handler(ExtractionError)
    async def extraction_error_handler(request, exc):
        return {"error": str(exc)}, 422
    
    @app.exception_handler(ProcessingError)
    async def processing_error_handler(request, exc):
        return {"error": str(exc)}, 500 