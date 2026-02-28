"""Hierarchical chunking strategy.

This chunker respects document structure by:
1. Keeping content within sections together when possible
2. Using section headings as natural chunk boundaries
3. Preserving the section hierarchy for context
4. Handling tables and images as atomic units
"""

from uuid import UUID

from src.ingestion.chunking.base import BaseChunker, Chunk, ChunkingResult
from src.ingestion.metadata import ContentType, ParsedDocument


class HierarchicalChunker(BaseChunker):
    """
    Chunks documents based on their hierarchical structure.

    This chunker:
    - Uses section headers as primary boundaries
    - Keeps content within sections together
    - Splits large sections while preserving context
    - Treats tables and images as atomic units
    - Prepends section hierarchy for context enrichment
    """

    @property
    def strategy_name(self) -> str:
        return "hierarchical"

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        include_heading_in_chunk: bool = True,
    ):
        """
        Initialize the hierarchical chunker.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum chunk size (smaller chunks are merged)
            include_heading_in_chunk: Whether to include the section heading in chunk content
        """
        super().__init__(chunk_size, chunk_overlap, min_chunk_size)
        self.include_heading_in_chunk = include_heading_in_chunk

    def chunk(self, document: ParsedDocument) -> ChunkingResult:
        """
        Chunk a parsed document using hierarchical structure.

        Args:
            document: The parsed document to chunk

        Returns:
            ChunkingResult containing all chunks and statistics
        """
        chunks: list[Chunk] = []
        document_id = document.metadata.document_id

        # Group content items by section
        sections = self._group_by_section(document.content_items)

        chunk_index = 0
        for section in sections:
            section_chunks = self._chunk_section(
                section=section,
                document_id=document_id,
                start_index=chunk_index,
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        # Compute statistics
        stats = self._compute_statistics(chunks)

        return ChunkingResult(
            document_id=document_id,
            chunks=chunks,
            total_chunks=len(chunks),
            chunking_strategy=self.strategy_name,
            **stats,
        )

    def _group_by_section(self, content_items: list[dict]) -> list[dict]:
        """
        Group content items by their section.

        Returns a list of sections, each with:
        - heading: section heading (or None for intro content)
        - hierarchy: section hierarchy path
        - items: list of content items in this section
        - page_numbers: set of pages this section spans
        """
        sections = []
        current_section = {
            "heading": None,
            "hierarchy": [],
            "items": [],
            "page_numbers": set(),
        }

        for item in content_items:
            item_type = item.get("type", "text")

            if item_type == "heading":
                # Save current section if it has content
                if current_section["items"]:
                    sections.append(current_section)

                # Start new section
                current_section = {
                    "heading": item.get("text", "").strip(),
                    "hierarchy": item.get("section_hierarchy", []),
                    "items": [],
                    "page_numbers": set(),
                }
            else:
                # Add item to current section
                current_section["items"].append(item)
                for page in item.get("page_numbers", []):
                    current_section["page_numbers"].add(page)

        # Don't forget the last section
        if current_section["items"]:
            sections.append(current_section)

        return sections

    def _chunk_section(
        self,
        section: dict,
        document_id: UUID,
        start_index: int,
    ) -> list[Chunk]:
        """
        Chunk a single section.

        Handles:
        - Small sections: keep as single chunk
        - Large sections: split while maintaining overlap
        - Tables/images: keep as atomic units
        """
        chunks = []
        heading = section.get("heading")
        hierarchy = section.get("hierarchy", [])
        items = section.get("items", [])
        section_pages = section.get("page_numbers", set())

        if not items:
            return chunks

        # Build content from items, handling different types
        current_content = []
        current_pages = set()
        current_has_table_or_image = False
        chunk_index = start_index

        def flush_chunk(content_type: ContentType = ContentType.TEXT):
            """Create a chunk from accumulated content."""
            nonlocal chunk_index

            if not current_content:
                return

            content = "\n\n".join(current_content)

            # Skip if too small (unless it's a table or image)
            # Use 25 as a hard floor so short-section docs (e.g. resumes) aren't silently dropped
            if len(content) < min(self.min_chunk_size, 25) and content_type == ContentType.TEXT:
                return

            # Prepend heading if configured
            if self.include_heading_in_chunk and heading:
                content = f"## {heading}\n\n{content}"

            chunk = self._create_chunk(
                content=content,
                document_id=document_id,
                chunk_index=chunk_index,
                page_numbers=sorted(current_pages) if current_pages else sorted(section_pages),
                content_type=content_type,
                section_hierarchy=hierarchy,
                heading=heading,
            )
            chunks.append(chunk)
            chunk_index += 1

        for item in items:
            item_type = item.get("type", "text")
            item_text = item.get("text", "").strip()
            item_pages = item.get("page_numbers", [])

            if not item_text:
                continue

            # Tables and images are atomic - create separate chunks
            if item_type == "table":
                # Flush any accumulated text first
                if current_content:
                    flush_chunk(
                        ContentType.MIXED if current_has_table_or_image else ContentType.TEXT
                    )
                    current_content = []
                    current_pages = set()
                    current_has_table_or_image = False

                # Create table chunk
                table_content = item_text
                if self.include_heading_in_chunk and heading:
                    table_content = f"## {heading}\n\n{table_content}"

                chunk = self._create_chunk(
                    content=table_content,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    page_numbers=item_pages if item_pages else sorted(section_pages),
                    content_type=ContentType.TABLE,
                    section_hierarchy=hierarchy,
                    heading=heading,
                )
                chunks.append(chunk)
                chunk_index += 1

            elif item_type == "image":
                # For images, we'd normally have a description from vision model
                # For now, create a placeholder or skip if no description
                if item_text:
                    if current_content:
                        flush_chunk(
                            ContentType.MIXED if current_has_table_or_image else ContentType.TEXT
                        )
                        current_content = []
                        current_pages = set()
                        current_has_table_or_image = False

                    image_content = f"[Image: {item_text}]"
                    if self.include_heading_in_chunk and heading:
                        image_content = f"## {heading}\n\n{image_content}"

                    chunk = self._create_chunk(
                        content=image_content,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        page_numbers=item_pages if item_pages else sorted(section_pages),
                        content_type=ContentType.IMAGE,
                        section_hierarchy=hierarchy,
                        heading=heading,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

            else:
                # Text or list items - accumulate until chunk size reached
                # Check if adding this would exceed chunk size
                potential_content = "\n\n".join(current_content + [item_text])

                if len(potential_content) > self.chunk_size and current_content:
                    # Flush current chunk and start new one
                    flush_chunk(
                        ContentType.MIXED if current_has_table_or_image else ContentType.TEXT
                    )
                    current_content = [item_text]
                    current_pages = set(item_pages)
                    current_has_table_or_image = False
                else:
                    current_content.append(item_text)
                    current_pages.update(item_pages)

        # Flush remaining content
        if current_content:
            flush_chunk(ContentType.MIXED if current_has_table_or_image else ContentType.TEXT)

        return chunks
