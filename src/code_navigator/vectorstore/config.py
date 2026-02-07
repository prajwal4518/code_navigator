"""
Configuration for vector storage.

Uses Pydantic Settings for type-safe configuration with:
- Environment variable loading
- .env file support
- Default values for local development
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorStoreSettings(BaseSettings):
    """Configuration for embeddings and vector storage.

    All settings can be overridden via environment variables.
    Pydantic Settings automatically loads from .env files.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Embedding model settings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model name for embeddings",
    )
    use_local_embeddings: bool = Field(
        default=True,
        description="Use local embeddings (True) or API embeddings (False)",
    )

    # ChromaDB settings
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma"),
        description="Directory for ChromaDB persistence",
    )
    chroma_collection_name: str = Field(
        default="code_navigator",
        description="Name of the ChromaDB collection",
    )

    # Batch processing settings
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation",
    )

    # Gemini settings (for API embeddings)
    google_api_key: str | None = Field(
        default=None,
        description="Google API key for Gemini embeddings",
    )


@lru_cache
def get_settings() -> VectorStoreSettings:
    """Get cached settings instance.

    Using lru_cache ensures we only load settings once.
    This is important because loading .env files is I/O.
    """
    return VectorStoreSettings()
