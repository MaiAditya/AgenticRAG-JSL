from loguru import logger
from src.extractors.image_extractor import ImageExtractor
from src.extractors.table_extractor import TableExtractor
from src.extractors.text_extractor import TextExtractor

_image_extractor = None
_table_extractor = None
_text_extractor = None

def initialize_models():
    """Initialize all ML models before server startup"""
    global _image_extractor, _table_extractor, _text_extractor
    
    logger.info("Initializing ML models...")
    
    try:
        logger.info("Loading Image Extractor...")
        _image_extractor = ImageExtractor()
        
        try:
            logger.info("Loading Table Extractor...")
            _table_extractor = TableExtractor()
        except ImportError as e:
            logger.warning(f"Table Extractor initialization failed: {str(e)}")
            logger.warning("Table extraction functionality will be disabled")
            _table_extractor = None
        
        logger.info("Loading Text Extractor...")
        _text_extractor = TextExtractor()
        
        logger.info("All available models initialized successfully!")
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise

def get_extractors():
    """Get initialized extractors"""
    extractors = {
        "image": _image_extractor,
        "text": _text_extractor
    }
    
    if _table_extractor:
        extractors["table"] = _table_extractor
        
    return {k: v for k, v in extractors.items() if v is not None} 