"""Simple JSON file-based document registry.

Tracks which documents are indexed so the API can list and delete them.
On startup the registry is seeded from ChromaDB, so documents indexed
via scripts (not the API) also appear in the UI.
"""

import json
import logging
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


class DocumentRegistry:
    """Thread-safe registry backed by a JSON file."""

    def __init__(self, path: str | Path = "data/processed/registry.json"):
        self.path = Path(path)
        self._lock = Lock()
        self._data: dict[str, dict] = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception as e:
                logger.warning(f"Could not load registry: {e}")
        return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))

    def add(self, doc_info: dict) -> None:
        with self._lock:
            self._data[doc_info["document_id"]] = doc_info
            self._save()

    def remove(self, document_id: str) -> None:
        with self._lock:
            self._data.pop(document_id, None)
            self._save()

    def get(self, document_id: str) -> dict | None:
        return self._data.get(document_id)

    def list(self) -> list[dict]:
        return list(self._data.values())

    def seed_from_chroma(self, vectordb) -> int:
        """
        Populate registry with documents that are in ChromaDB but not yet
        in the registry file. Called at API startup so CLI-indexed documents
        appear in the UI immediately.

        Returns the number of new entries added.
        """
        try:
            all_items = vectordb.collection.get(include=["metadatas"])
        except Exception as e:
            logger.warning(f"Could not seed registry from ChromaDB: {e}")
            return 0

        # Aggregate per document
        docs: dict[str, dict] = {}
        for meta in all_items.get("metadatas", []):
            doc_id = meta.get("document_id")
            if not doc_id:
                continue
            if doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta.get("filename", ""),
                    "title": meta.get("title", ""),
                    "total_chunks": 0,
                    "text_chunks": 0,
                    "table_chunks": 0,
                    "image_chunks": 0,
                    "indexed_at": "imported",
                }
            docs[doc_id]["total_chunks"] += 1
            ct = meta.get("content_type", "text")
            if ct == "text":
                docs[doc_id]["text_chunks"] += 1
            elif ct == "table":
                docs[doc_id]["table_chunks"] += 1
            elif ct == "image":
                docs[doc_id]["image_chunks"] += 1

        added = 0
        for doc_id, info in docs.items():
            if not self.get(doc_id):
                self.add(info)
                added += 1

        if added:
            logger.info(f"Registry: seeded {added} documents from ChromaDB")
        return added
