"""Chunk retrieval for the RAG pipeline."""

import logging
from dataclasses import dataclass

from src.embeddings.bedrock_embeddings import BedrockEmbeddings
from src.vectordb.chroma_client import ChromaVectorDB

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its relevance score and source metadata."""

    chunk_id: str
    document_id: str
    filename: str
    title: str
    content: str
    content_type: str
    heading: str
    section_hierarchy: list[str]
    page_numbers: list[int]
    chunk_index: int
    score: float  # cosine similarity [0, 1], higher = more relevant

    def format_citation(self) -> str:
        """Short citation string suitable for display."""
        pages = ", ".join(f"p.{p}" for p in self.page_numbers) if self.page_numbers else "?"
        if self.heading:
            return f"{self.filename} — {self.heading} ({pages})"
        return f"{self.filename} ({pages})"

    def format_context(self) -> str:
        """Format this chunk as context for an LLM prompt."""
        lines = []
        if self.section_hierarchy:
            lines.append(f"[Source: {' > '.join(self.section_hierarchy)}]")
        lines.append(self.content)
        return "\n".join(lines)


class Retriever:
    """
    Retrieves the most relevant chunks for a natural-language query.

    Flow:
    1. Embed the query with Bedrock Titan Embed v2
    2. Search ChromaDB for top-k nearest chunks (cosine similarity)
    3. Optionally filter by document or content type
    4. Return structured RetrievedChunk objects sorted by score
    """

    def __init__(
        self,
        embeddings: BedrockEmbeddings | None = None,
        vectordb: ChromaVectorDB | None = None,
    ):
        self.embeddings = embeddings or BedrockEmbeddings()
        self.vectordb = vectordb or ChromaVectorDB()

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
        content_type: str | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: Natural-language question or search string.
            n_results: Maximum number of chunks to return.
            document_id: Restrict retrieval to a specific document.
            document_ids: Restrict retrieval to a list of documents.
            content_type: Filter by "text", "table", "image", or "mixed".
            min_score: Drop chunks with cosine similarity below this threshold.

        Returns:
            List of RetrievedChunk, sorted by score descending.
        """
        query_embedding = self.embeddings.embed(query)

        raw = self.vectordb.query(
            query_embedding=query_embedding,
            n_results=n_results,
            document_id=document_id,
            document_ids=document_ids,
            content_type=content_type,
        )

        chunks = []
        for r in raw:
            if r["score"] < min_score:
                continue
            m = r["metadata"]
            chunks.append(RetrievedChunk(
                chunk_id=r["id"],
                document_id=m.get("document_id", ""),
                filename=m.get("filename", ""),
                title=m.get("title", ""),
                content=r["document"],
                content_type=m.get("content_type", "text"),
                heading=m.get("heading", ""),
                section_hierarchy=m.get("section_hierarchy", []),
                page_numbers=m.get("page_numbers", []),
                chunk_index=m.get("chunk_index", 0),
                score=r["score"],
            ))

        return chunks
