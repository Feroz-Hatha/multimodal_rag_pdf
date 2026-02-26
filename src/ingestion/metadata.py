"""Metadata models for documents and chunks."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Type of content in a chunk."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    MIXED = "mixed"


class DocumentMetadata(BaseModel):
    """Metadata for a parsed document."""

    document_id: UUID = Field(default_factory=uuid4)
    filename: str
    title: str | None = None
    total_pages: int
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    file_hash: str  # SHA256 hash for deduplication
    file_size_bytes: int

    # Extracted document info
    author: str | None = None
    creation_date: str | None = None

    # Processing info
    docling_version: str | None = None
    processing_time_seconds: float | None = None

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }


class ChunkMetadata(BaseModel):
    """Metadata for a single chunk."""

    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    chunk_index: int  # Position in document (0-indexed)

    # Location info
    page_numbers: list[int]  # Pages this chunk spans

    # Content classification
    content_type: ContentType = ContentType.TEXT

    # Hierarchical context
    section_hierarchy: list[str] = Field(default_factory=list)  # e.g., ["Chapter 1", "Section 1.2", "1.2.1 Methods"]
    heading: str | None = None  # Immediate section heading

    # For hierarchical chunking
    parent_chunk_id: UUID | None = None
    depth: int = 0  # 0 = top-level

    # Size metrics
    char_count: int
    token_count_estimate: int  # Rough estimate: chars / 4

    # For tables
    table_id: str | None = None
    table_caption: str | None = None

    # For images
    image_id: str | None = None
    image_caption: str | None = None
    image_path: str | None = None  # Path to extracted image file

    class Config:
        json_encoders = {
            UUID: str,
        }

    def get_context_prefix(self) -> str:
        """Generate a context prefix from section hierarchy."""
        if not self.section_hierarchy:
            return ""
        return " > ".join(self.section_hierarchy)


class ParsedDocument(BaseModel):
    """A fully parsed document with its content and metadata."""

    metadata: DocumentMetadata

    # Raw content in different formats
    markdown_content: str  # Full document as markdown

    # Structured elements
    tables: list[dict[str, Any]] = Field(default_factory=list)
    images: list[dict[str, Any]] = Field(default_factory=list)

    # Section structure
    sections: list[dict[str, Any]] = Field(default_factory=list)

    # All content items in order (for chunking)
    # Each item has: type, text, level, page_numbers, section_hierarchy
    content_items: list[dict[str, Any]] = Field(default_factory=list)

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }
