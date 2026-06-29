from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # model_config = SettingsConfigDict(
    #     env_file=".env",
    #     extra="ignore"
    # )
    groq_api_key: str
    groq_model: str = "llama-3.1-8b-instant"
    # groq_model: str = "llama-3.3-70b-versatile"
    openai_api_key: str | None = None
    pinecone_api_key: str | None = None  # <- new
    embedding_model: str = "text-embedding-3-small"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    qdrant_url: str = "https://e9f82bec-cb9a-4d77-a665-4e15fbd37436.us-west-2-0.aws.cloud.qdrant.io"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "goutam_test"
    chunk_size: int = 600
    chunk_overlap: int = 100

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()


