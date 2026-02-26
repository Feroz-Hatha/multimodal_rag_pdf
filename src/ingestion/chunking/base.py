"""Base chunker interface and common data structures."""

from abc import ABC, abstractmethod
from uuid import UUID

from pydantic import BaseModel, Field

from src.ingestion.metadata import ChunkMetadata, ContentType, ParsedDocument


class Chunk(BaseModel):
    """A chunk of content with its metadata."""

    content: str  # The actual text content
    metadata: ChunkMetadata

    # Context-enriched content (content with section hierarchy prepended)
    enriched_content: str | None = None

    def get_embedding_text(self) -> str:
        """Get the text to use for embedding generation."""
        if self.enriched_content:
            return self.enriched_content

        # Prepend context if available
        context = self.metadata.get_context_prefix()
        if context:
            return f"[{context}]\n\n{self.content}"
        return self.content

    class Config:
        json_encoders = {
            UUID: str,
        }


class ChunkingResult(BaseModel):
    """Result of chunking a document."""

    document_id: UUID
    chunks: list[Chunk]
    total_chunks: int
    chunking_strategy: str

    # Statistics
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int

    # By content type
    text_chunks: int = 0
    table_chunks: int = 0
    image_chunks: int = 0
    mixed_chunks: int = 0

    class Config:
        json_encoders = {
            UUID: str,
        }


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum chunk size (smaller chunks are merged)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the name of this chunking strategy."""
        pass

    @abstractmethod
    def chunk(self, document: ParsedDocument) -> ChunkingResult:
        """
        Chunk a parsed document.

        Args:
            document: The parsed document to chunk

        Returns:
            ChunkingResult containing all chunks and statistics
        """
        pass

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 chars)."""
        return len(text) // 4

    def _create_chunk(
        self,
        content: str,
        document_id: UUID,
        chunk_index: int,
        page_numbers: list[int],
        content_type: ContentType = ContentType.TEXT,
        section_hierarchy: list[str] | None = None,
        heading: str | None = None,
        parent_chunk_id: UUID | None = None,
        depth: int = 0,
        **kwargs,
    ) -> Chunk:
        """
        Create a chunk with proper metadata.

        Args:
            content: The chunk content
            document_id: Parent document ID
            chunk_index: Position in document
            page_numbers: Pages this chunk spans
            content_type: Type of content
            section_hierarchy: List of parent section headings
            heading: Immediate section heading
            parent_chunk_id: For hierarchical chunking
            depth: Nesting depth
            **kwargs: Additional metadata fields

        Returns:
            A Chunk instance
        """
        metadata = ChunkMetadata(
            document_id=document_id,
            chunk_index=chunk_index,
            page_numbers=page_numbers,
            content_type=content_type,
            section_hierarchy=section_hierarchy or [],
            heading=heading,
            parent_chunk_id=parent_chunk_id,
            depth=depth,
            char_count=len(content),
            token_count_estimate=self._estimate_tokens(content),
            **kwargs,
        )

        # Create enriched content with context
        context_prefix = metadata.get_context_prefix()
        enriched = f"[{context_prefix}]\n\n{content}" if context_prefix else content

        return Chunk(
            content=content,
            metadata=metadata,
            enriched_content=enriched,
        )

    def _compute_statistics(self, chunks: list[Chunk]) -> dict:
        """Compute statistics for chunking result."""
        if not chunks:
            return {
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "text_chunks": 0,
                "table_chunks": 0,
                "image_chunks": 0,
                "mixed_chunks": 0,
            }

        sizes = [c.metadata.char_count for c in chunks]
        type_counts = {
            ContentType.TEXT: 0,
            ContentType.TABLE: 0,
            ContentType.IMAGE: 0,
            ContentType.MIXED: 0,
        }
        for c in chunks:
            type_counts[c.metadata.content_type] += 1

        return {
            "avg_chunk_size": sum(sizes) / len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "text_chunks": type_counts[ContentType.TEXT],
            "table_chunks": type_counts[ContentType.TABLE],
            "image_chunks": type_counts[ContentType.IMAGE],
            "mixed_chunks": type_counts[ContentType.MIXED],
        }
