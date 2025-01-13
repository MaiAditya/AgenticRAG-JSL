from pydantic import BaseSettings

class Settings(BaseSettings):
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    class Config:
        env_file = ".env"

settings = Settings() 