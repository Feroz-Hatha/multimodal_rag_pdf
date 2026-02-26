"""ChromaDB vector database client."""

import json
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.ingestion.chunking.base import Chunk

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "pdf_chunks"


class ChromaVectorDB:
    """
    Persistent ChromaDB client for storing and querying document chunks.

    Uses cosine distance (HNSW index). Each chunk is stored with:
    - ID: chunk UUID
    - Document: enriched text used for embedding
    - Embedding: Titan Embed vector (1024-dim)
    - Metadata: flat dict of chunk + document fields
      (page_numbers and section_hierarchy are JSON-serialised since
      ChromaDB requires flat str/int/float/bool metadata values)
    """

    def __init__(
        self,
        persist_dir: str | Path = "data/chroma_db",
        collection_name: str = _COLLECTION_NAME,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB ready — collection '{collection_name}' "
            f"has {self.collection.count()} chunks"
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def _flatten_metadata(self, chunk: Chunk, doc_meta: dict) -> dict:
        """
        Build a flat metadata dict for ChromaDB.

        doc_meta must contain: filename, title, file_hash
        """
        m = chunk.metadata
        return {
            # Document-level
            "document_id": str(m.document_id),
            "filename": doc_meta["filename"],
            "title": doc_meta.get("title", ""),
            "file_hash": doc_meta["file_hash"],
            # Chunk-level
            "chunk_id": str(m.chunk_id),
            "chunk_index": m.chunk_index,
            "content_type": m.content_type.value,
            "heading": m.heading or "",
            # Serialised lists
            "page_numbers": json.dumps(m.page_numbers),
            "section_hierarchy": json.dumps(m.section_hierarchy),
            # Size metrics
            "char_count": m.char_count,
            "token_count_estimate": m.token_count_estimate,
        }

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        doc_meta: dict,
    ) -> None:
        """
        Upsert chunks into the collection.

        Using upsert means re-indexing the same document is safe — existing
        chunks (same chunk_id) are overwritten rather than duplicated.

        doc_meta: {"filename": ..., "title": ..., "file_hash": ...}
        """
        if not chunks:
            return

        self.collection.upsert(
            ids=[str(c.metadata.chunk_id) for c in chunks],
            embeddings=embeddings,
            documents=[c.get_embedding_text() for c in chunks],
            metadatas=[self._flatten_metadata(c, doc_meta) for c in chunks],
        )
        logger.info(f"Stored {len(chunks)} chunks for '{doc_meta['filename']}'")

    # ------------------------------------------------------------------
    # Deduplication helpers
    # ------------------------------------------------------------------

    def file_hash_exists(self, file_hash: str) -> str | None:
        """
        Return the document_id if a file with this SHA256 hash is already
        indexed, otherwise None.
        """
        results = self.client.get_collection(self.collection.name).get(
            where={"file_hash": {"$eq": file_hash}},
            limit=1,
            include=["metadatas"],
        )
        if results["ids"]:
            return results["metadatas"][0]["document_id"]
        return None

    def document_exists(self, document_id: str) -> bool:
        """Check whether any chunks for this document_id exist."""
        results = self.collection.get(
            where={"document_id": {"$eq": document_id}},
            limit=1,
            include=[],
        )
        return bool(results["ids"])

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
        content_type: str | None = None,
    ) -> list[dict]:
        """
        Return the top-n most similar chunks for a query embedding.

        Optional filters:
        - document_ids: restrict to a list of documents (uses $in operator)
        - document_id: restrict to a single document (ignored when document_ids is set)
        - content_type: one of "text", "table", "image", "mixed"

        Each result dict has: id, document, metadata, distance, score
        """
        total = self.collection.count()
        if total == 0:
            return []

        # Build filter conditions
        conditions: list[dict] = []
        if document_ids is not None:
            if not document_ids:
                return []  # empty list → no results
            conditions.append({"document_id": {"$in": document_ids}})
        elif document_id:
            conditions.append({"document_id": {"$eq": document_id}})
        if content_type:
            conditions.append({"content_type": {"$eq": content_type}})

        if len(conditions) == 0:
            where = None
        elif len(conditions) == 1:
            where = conditions[0]
        else:
            where = {"$and": conditions}

        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for i, chunk_id in enumerate(raw["ids"][0]):
            meta = dict(raw["metadatas"][0][i])
            # Deserialise JSON-encoded lists
            meta["page_numbers"] = json.loads(meta["page_numbers"])
            meta["section_hierarchy"] = json.loads(meta["section_hierarchy"])
            distance = raw["distances"][0][i]
            results.append({
                "id": chunk_id,
                "document": raw["documents"][0][i],
                "metadata": meta,
                "distance": distance,
                "score": 1.0 - distance,  # cosine similarity
            })

        return results

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        existing = self.collection.get(
            where={"document_id": {"$eq": document_id}},
            include=[],
        )
        count = len(existing["ids"])
        if count:
            self.collection.delete(ids=existing["ids"])
            logger.info(f"Deleted {count} chunks for document_id={document_id}")
        return count

    def get_stats(self) -> dict:
        """Basic collection statistics."""
        return {
            "total_chunks": self.collection.count(),
            "persist_dir": str(self.persist_dir),
        }
