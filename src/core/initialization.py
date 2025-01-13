from loguru import logger
from src.extractors.image_extractor import ImageExtractor
from src.extractors.table_extractor import TableExtractor
from src.extractors.text_extractor import TextExtractor
from typing import Dict, Any

_extractors = {}
_initialized = False
_models_initialized = False

def initialize_models() -> None:
    """Initialize all ML models"""
    global _models_initialized
    
    if _models_initialized:
        return
        
    logger.info("Initializing ML models...")
    try:
        # Initialize any ML models here if needed
        # For now, this is a placeholder for future model initialization
        _models_initialized = True
        logger.info("ML models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing ML models: {str(e)}")
        raise

def initialize_extractors() -> None:
    """Initialize all extractors as singletons"""
    global _extractors, _initialized
    
    if _initialized:
        return
    
    logger.info("Initializing extractors...")
    
    try:
        _extractors["text"] = TextExtractor()
        logger.info("Text extractor initialized")
        
        _extractors["image"] = ImageExtractor()
        logger.info("Image extractor initialized")
        
        try:
            _extractors["table"] = TableExtractor()
            logger.info("Table extractor initialized")
        except ImportError as e:
            logger.warning(f"Table extractor initialization failed: {str(e)}")
            logger.warning("Table extraction functionality will be disabled")
    
        _initialized = True
        logger.info("All extractors initialized successfully!")
    except Exception as e:
        logger.error(f"Error initializing extractors: {str(e)}")
        raise

def get_extractors() -> Dict[str, Any]:
    """Get initialized extractors"""
    global _extractors, _initialized
    
    if not _initialized:
        initialize_extractors()
    
    return _extractors 