"""Semantic chunking strategy.

This chunker creates chunks based on semantic similarity:
1. Splits text into sentences
2. Computes embeddings for sentence groups
3. Identifies semantic breakpoints where topic shifts
4. Creates chunks at natural semantic boundaries
"""

import logging
import re
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.ingestion.chunking.base import BaseChunker, Chunk, ChunkingResult
from src.ingestion.metadata import ContentType, ParsedDocument

logger = logging.getLogger(__name__)


class SemanticChunker(BaseChunker):
    """
    Chunks documents based on semantic similarity.

    This chunker:
    - Splits content into sentences
    - Uses embeddings to detect topic shifts
    - Creates chunks at semantic boundaries
    - Falls back to size-based splitting for very long sections
    """

    @property
    def strategy_name(self) -> str:
        return "semantic"

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        buffer_size: int = 3,
    ):
        """
        Initialize the semantic chunker.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum chunk size (smaller chunks are merged)
            model_name: Sentence transformer model for embeddings
            similarity_threshold: Threshold below which to create a new chunk
            buffer_size: Number of sentences to consider on each side for similarity
        """
        super().__init__(chunk_size, chunk_overlap, min_chunk_size)
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.buffer_size = buffer_size
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def chunk(self, document: ParsedDocument) -> ChunkingResult:
        """
        Chunk a parsed document using semantic similarity.

        Args:
            document: The parsed document to chunk

        Returns:
            ChunkingResult containing all chunks and statistics
        """
        chunks: list[Chunk] = []
        document_id = document.metadata.document_id

        # Process content items
        chunk_index = 0

        # Group items by type for appropriate handling
        text_buffer = []
        text_pages = set()
        current_hierarchy = []
        current_heading = None

        for item in document.content_items:
            item_type = item.get("type", "text")
            item_text = item.get("text", "").strip()
            item_pages = item.get("page_numbers", [])
            item_hierarchy = item.get("section_hierarchy", [])

            if not item_text:
                continue

            # Track section context
            if item_type == "heading":
                # Flush accumulated text before section change
                if text_buffer:
                    text_chunks = self._chunk_text_semantically(
                        text="\n\n".join(text_buffer),
                        document_id=document_id,
                        start_index=chunk_index,
                        page_numbers=sorted(text_pages),
                        section_hierarchy=current_hierarchy,
                        heading=current_heading,
                    )
                    chunks.extend(text_chunks)
                    chunk_index += len(text_chunks)
                    text_buffer = []
                    text_pages = set()

                current_heading = item_text
                current_hierarchy = item_hierarchy
                continue

            # Tables are atomic
            if item_type == "table":
                # Flush text first
                if text_buffer:
                    text_chunks = self._chunk_text_semantically(
                        text="\n\n".join(text_buffer),
                        document_id=document_id,
                        start_index=chunk_index,
                        page_numbers=sorted(text_pages),
                        section_hierarchy=current_hierarchy,
                        heading=current_heading,
                    )
                    chunks.extend(text_chunks)
                    chunk_index += len(text_chunks)
                    text_buffer = []
                    text_pages = set()

                # Create table chunk
                table_content = item_text
                if current_heading:
                    table_content = f"## {current_heading}\n\n{table_content}"

                chunk = self._create_chunk(
                    content=table_content,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    page_numbers=item_pages,
                    content_type=ContentType.TABLE,
                    section_hierarchy=current_hierarchy,
                    heading=current_heading,
                )
                chunks.append(chunk)
                chunk_index += 1

            # Images are atomic
            elif item_type == "image":
                if item_text:
                    # Flush text first
                    if text_buffer:
                        text_chunks = self._chunk_text_semantically(
                            text="\n\n".join(text_buffer),
                            document_id=document_id,
                            start_index=chunk_index,
                            page_numbers=sorted(text_pages),
                            section_hierarchy=current_hierarchy,
                            heading=current_heading,
                        )
                        chunks.extend(text_chunks)
                        chunk_index += len(text_chunks)
                        text_buffer = []
                        text_pages = set()

                    image_content = f"[Image: {item_text}]"
                    if current_heading:
                        image_content = f"## {current_heading}\n\n{image_content}"

                    chunk = self._create_chunk(
                        content=image_content,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        page_numbers=item_pages,
                        content_type=ContentType.IMAGE,
                        section_hierarchy=current_hierarchy,
                        heading=current_heading,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

            # Text and list items - accumulate for semantic chunking
            else:
                text_buffer.append(item_text)
                text_pages.update(item_pages)
                if item_hierarchy:
                    current_hierarchy = item_hierarchy

        # Flush remaining text
        if text_buffer:
            text_chunks = self._chunk_text_semantically(
                text="\n\n".join(text_buffer),
                document_id=document_id,
                start_index=chunk_index,
                page_numbers=sorted(text_pages),
                section_hierarchy=current_hierarchy,
                heading=current_heading,
            )
            chunks.extend(text_chunks)

        # Compute statistics
        stats = self._compute_statistics(chunks)

        return ChunkingResult(
            document_id=document_id,
            chunks=chunks,
            total_chunks=len(chunks),
            chunking_strategy=self.strategy_name,
            **stats,
        )

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting - handles common cases
        # Split on period, question mark, exclamation followed by space and capital
        # Also handle newlines as sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*(?=\S)', text)

        # Clean up and filter empty
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _compute_similarities(self, sentences: list[str]) -> list[float]:
        """
        Compute similarity between adjacent sentence groups.

        Uses a sliding window approach where we compare:
        - The embedding of sentences [i-buffer : i]
        - The embedding of sentences [i : i+buffer]

        Returns a list of similarity scores for each potential breakpoint.
        """
        if len(sentences) <= 2 * self.buffer_size:
            return []

        # Combine sentences into groups for embedding
        groups = []
        for i in range(self.buffer_size, len(sentences) - self.buffer_size + 1):
            before = " ".join(sentences[max(0, i - self.buffer_size):i])
            after = " ".join(sentences[i:min(len(sentences), i + self.buffer_size)])
            groups.append((before, after))

        if not groups:
            return []

        # Compute embeddings for all groups at once
        all_texts = []
        for before, after in groups:
            all_texts.extend([before, after])

        embeddings = self.model.encode(all_texts, convert_to_numpy=True)

        # Calculate cosine similarities between before/after pairs
        similarities = []
        for i in range(0, len(embeddings), 2):
            before_emb = embeddings[i]
            after_emb = embeddings[i + 1]
            similarity = np.dot(before_emb, after_emb) / (
                np.linalg.norm(before_emb) * np.linalg.norm(after_emb)
            )
            similarities.append(float(similarity))

        return similarities

    def _find_breakpoints(self, similarities: list[float], sentences: list[str]) -> list[int]:
        """
        Find semantic breakpoints based on similarity scores.

        A breakpoint is where similarity drops below threshold,
        adjusted to respect chunk size constraints.
        """
        breakpoints = []

        # Track cumulative character count
        current_length = 0
        last_break = 0

        for i, sim in enumerate(similarities):
            # Adjust index to account for buffer offset
            sentence_idx = i + self.buffer_size

            # Add length of sentences since last break
            for j in range(last_break, sentence_idx):
                if j < len(sentences):
                    current_length += len(sentences[j])

            # Check if we should break
            should_break = False

            # Break if similarity is low (topic shift)
            if sim < self.similarity_threshold:
                should_break = True

            # Force break if we've accumulated too much content
            if current_length >= self.chunk_size:
                should_break = True

            if should_break and current_length >= self.min_chunk_size:
                breakpoints.append(sentence_idx)
                current_length = 0
                last_break = sentence_idx

        return breakpoints

    def _chunk_text_semantically(
        self,
        text: str,
        document_id,
        start_index: int,
        page_numbers: list[int],
        section_hierarchy: list[str],
        heading: str | None,
    ) -> list[Chunk]:
        """
        Chunk text using semantic similarity.

        Falls back to simple size-based splitting if text is too short
        for meaningful semantic analysis.
        """
        chunks = []

        # Skip empty text
        if not text.strip():
            return chunks

        sentences = self._split_into_sentences(text)

        # If too few sentences, fall back to simple chunking
        if len(sentences) <= 2 * self.buffer_size:
            return self._simple_chunk(
                text=text,
                document_id=document_id,
                start_index=start_index,
                page_numbers=page_numbers,
                section_hierarchy=section_hierarchy,
                heading=heading,
            )

        # Compute similarities and find breakpoints
        similarities = self._compute_similarities(sentences)
        breakpoints = self._find_breakpoints(similarities, sentences)

        # Create chunks based on breakpoints
        chunk_index = start_index
        prev_break = 0

        for bp in breakpoints:
            chunk_sentences = sentences[prev_break:bp]
            if chunk_sentences:
                content = " ".join(chunk_sentences)
                if heading:
                    content = f"## {heading}\n\n{content}"

                chunk = self._create_chunk(
                    content=content,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    page_numbers=page_numbers,
                    content_type=ContentType.TEXT,
                    section_hierarchy=section_hierarchy,
                    heading=heading,
                )
                chunks.append(chunk)
                chunk_index += 1

            prev_break = bp

        # Handle remaining sentences
        if prev_break < len(sentences):
            chunk_sentences = sentences[prev_break:]
            if chunk_sentences:
                content = " ".join(chunk_sentences)
                if heading:
                    content = f"## {heading}\n\n{content}"

                chunk = self._create_chunk(
                    content=content,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    page_numbers=page_numbers,
                    content_type=ContentType.TEXT,
                    section_hierarchy=section_hierarchy,
                    heading=heading,
                )
                chunks.append(chunk)

        return chunks

    def _simple_chunk(
        self,
        text: str,
        document_id,
        start_index: int,
        page_numbers: list[int],
        section_hierarchy: list[str],
        heading: str | None,
    ) -> list[Chunk]:
        """
        Simple size-based chunking fallback.

        Used when text is too short for semantic analysis.
        """
        chunks = []

        # If short enough, keep as single chunk
        if len(text) <= self.chunk_size:
            content = text
            if heading:
                content = f"## {heading}\n\n{content}"

            chunk = self._create_chunk(
                content=content,
                document_id=document_id,
                chunk_index=start_index,
                page_numbers=page_numbers,
                content_type=ContentType.TEXT,
                section_hierarchy=section_hierarchy,
                heading=heading,
            )
            chunks.append(chunk)
            return chunks

        # Split by size with overlap
        chunk_index = start_index
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end near the target
                search_start = max(start + self.chunk_size - 100, start)
                search_end = min(start + self.chunk_size + 100, len(text))
                search_text = text[search_start:search_end]

                # Find sentence boundary
                for pattern in ['. ', '.\n', '? ', '?\n', '! ', '!\n']:
                    idx = search_text.find(pattern)
                    if idx != -1:
                        end = search_start + idx + len(pattern)
                        break

            content = text[start:end].strip()
            if heading:
                content = f"## {heading}\n\n{content}"

            if content:
                chunk = self._create_chunk(
                    content=content,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    page_numbers=page_numbers,
                    content_type=ContentType.TEXT,
                    section_hierarchy=section_hierarchy,
                    heading=heading,
                )
                chunks.append(chunk)
                chunk_index += 1

            start = end - self.chunk_overlap

        return chunks
