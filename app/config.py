from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    APP_NAME: str = "AI Knowledge Assistant"
    APP_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 5
    CONFIDENCE_THRESHOLD: float = 0.5
    
    VECTOR_DB_PATH: str = "./vector_db"
    DOCUMENTS_PATH: str = "./data/documents"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

settings = Settings()

os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
os.makedirs(settings.DOCUMENTS_PATH, exist_ok=True)