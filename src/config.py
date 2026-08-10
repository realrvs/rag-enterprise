"""
Application configuration using Pydantic Settings.
All environment variables are loaded from .env file.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    # ==========================================
    # LLM Configuration
    # ==========================================
    llm_model: str = "qwen2.5-coder:7b"
    llm_use_ollama: bool = True
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512
    ollama_url: str = "http://localhost:11434"

    # ==========================================
    # OpenAI / LLM
    # ==========================================
    openai_api_key: str
    openai_model: str = "gpt-4-turbo-preview"
    openai_embedding_model: str = "text-embedding-3-small"
    
    # ==========================================
    # Qdrant
    # ==========================================
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_collection: str = "enterprise_docs"
    qdrant_api_key: Optional[str] = None
    
    # ==========================================
    # Redis (Cache)
    # ==========================================
    redis_url: str = "redis://localhost:6379"
    cache_threshold: float = 0.92
    cache_ttl: int = 3600  # seconds
    
    # ==========================================
    # Observability
    # ==========================================
    langchain_api_key: Optional[str] = None
    langchain_project: str = "rag-enterprise"
    langchain_tracing_v2: bool = True
    
    # ==========================================
    # Application
    # ==========================================
    log_level: str = "INFO"
    environment: str = "development"
    secret_key: str = "change-me-in-production"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )
    
    @property
    def qdrant_url(self) -> str:
        """Get Qdrant REST API URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"
    
    @property
    def qdrant_grpc_url(self) -> str:
        """Get Qdrant gRPC API URL."""
        return f"{self.qdrant_host}:{self.qdrant_grpc_port}"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"


# Global settings instance
settings = Settings()