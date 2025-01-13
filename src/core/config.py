from pydantic_settings import BaseSettings, SettingsConfigDict
import torch

class Settings(BaseSettings):
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    ALLOWED_ORIGINS: list = ["http://localhost:3000"]
    OPENAI_API_KEY: str = ""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow"
    )

settings = Settings() 