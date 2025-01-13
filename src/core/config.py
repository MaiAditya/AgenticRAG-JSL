from pydantic_settings import BaseSettings, SettingsConfigDict
import torch
from loguru import logger
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(verbose=True)
logger.info("Loading environment variables from .env file")

class Settings(BaseSettings):
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOWED_ORIGINS: list = ["http://localhost:3000"]
    OPENAI_API_KEY: str
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )

    def __init__(self):
        super().__init__()
        # Validate OPENAI_API_KEY is set
        if not self.OPENAI_API_KEY:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY not found in environment variables or .env file")
                raise ValueError("OPENAI_API_KEY must be set in environment variables or .env file")
            self.OPENAI_API_KEY = api_key
        logger.info("Successfully loaded OPENAI_API_KEY from environment")

settings = Settings() 