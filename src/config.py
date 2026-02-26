"""Configuration management for PDF Multimodal RAG."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    upload_dir: Path = Field(default=Path("data/uploads"))
    processed_dir: Path = Field(default=Path("data/processed"))
    chroma_db_dir: Path = Field(default=Path("data/chroma_db"))

    # AWS Configuration
    aws_region: str = Field(default="us-east-1")
    aws_access_key_id: str | None = Field(default=None)
    aws_secret_access_key: str | None = Field(default=None)

    # AWS Bedrock
    bedrock_embedding_model_id: str = Field(default="amazon.titan-embed-text-v2:0")
    bedrock_llm_model_id: str = Field(default="us.anthropic.claude-3-5-sonnet-20241022-v2:0")

    # S3
    s3_bucket_name: str | None = Field(default=None)

    # Chunking Settings
    chunk_size: int = Field(default=1000, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks in characters")
    min_chunk_size: int = Field(default=100, description="Minimum chunk size in characters")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Resolve paths relative to base_dir if they're relative
        if not self.upload_dir.is_absolute():
            self.upload_dir = self.base_dir / self.upload_dir
        if not self.processed_dir.is_absolute():
            self.processed_dir = self.base_dir / self.processed_dir
        if not self.chroma_db_dir.is_absolute():
            self.chroma_db_dir = self.base_dir / self.chroma_db_dir

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
