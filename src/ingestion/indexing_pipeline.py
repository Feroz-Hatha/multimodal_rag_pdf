"""End-to-end indexing pipeline: PDF → Chunks → Embeddings → ChromaDB."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from src.embeddings.bedrock_embeddings import BedrockEmbeddings
from src.generation.image_describer import ImageDescriber
from src.ingestion.chunking.hierarchical import HierarchicalChunker
from src.ingestion.pdf_parser import PDFParser
from src.vectordb.chroma_client import ChromaVectorDB

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """Outcome of indexing a single document."""

    filename: str
    document_id: str
    total_chunks: int
    text_chunks: int
    table_chunks: int
    image_chunks: int
    processing_time_seconds: float = 0.0
    skipped: bool = False
    skip_reason: str = ""
    error: str = ""


class IndexingPipeline:
    """
    Orchestrates the full ingestion pipeline:

    1. Parse PDF with Docling (image bytes captured when present)
    2. Describe images with Claude Vision via Bedrock (optional)
    3. Chunk with HierarchicalChunker
    4. Generate embeddings with AWS Bedrock Titan Embed v2
    5. Upsert into ChromaDB

    Deduplication: if a file with the same SHA256 hash is already in
    ChromaDB the document is skipped unless force_reindex=True.
    """

    def __init__(
        self,
        parser: PDFParser | None = None,
        chunker: HierarchicalChunker | None = None,
        embeddings: BedrockEmbeddings | None = None,
        vectordb: ChromaVectorDB | None = None,
        image_describer: ImageDescriber | None = None,
        describe_images: bool = True,
        processed_dir: str | Path = "data/processed",
        save_parsed: bool = True,
    ):
        self.parser = parser or PDFParser()
        self.chunker = chunker or HierarchicalChunker()
        self.embeddings = embeddings or BedrockEmbeddings()
        self.vectordb = vectordb or ChromaVectorDB()
        self.image_describer = image_describer or (ImageDescriber() if describe_images else None)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.save_parsed = save_parsed

    def index_document(
        self,
        file_path: str | Path,
        force_reindex: bool = False,
    ) -> IndexingResult:
        """
        Index a single PDF document.

        Args:
            file_path: Path to the PDF file.
            force_reindex: Re-index even if the file hash already exists.

        Returns:
            IndexingResult with stats.
        """
        file_path = Path(file_path)
        start = time.time()

        # --- Parse ---
        logger.info(f"Parsing: {file_path.name}")
        parsed = self.parser.parse(file_path)
        doc_id = str(parsed.metadata.document_id)

        # --- Deduplication ---
        if not force_reindex:
            existing_id = self.vectordb.file_hash_exists(parsed.metadata.file_hash)
            if existing_id:
                logger.info(
                    f"'{file_path.name}' already indexed (doc_id={existing_id}) — skipping"
                )
                return IndexingResult(
                    filename=file_path.name,
                    document_id=existing_id,
                    total_chunks=0,
                    text_chunks=0,
                    table_chunks=0,
                    image_chunks=0,
                    skipped=True,
                    skip_reason="already_indexed",
                )

        # --- Image descriptions (before chunking so image chunks get text) ---
        if self.image_describer:
            n_images = sum(
                1 for item in parsed.content_items
                if item.get("type") == "image" and item.get("image_data")
            )
            if n_images:
                logger.info(f"Describing {n_images} images for: {file_path.name}")
                described = self.image_describer.describe_document_images(parsed.content_items)
                logger.info(f"Successfully described {described}/{n_images} images")

        # Strip any remaining image_data before JSON serialisation
        for item in parsed.content_items:
            item.pop("image_data", None)

        # --- Chunk ---
        logger.info(f"Chunking: {file_path.name}")
        result = self.chunker.chunk(parsed)
        chunks = result.chunks

        if not chunks:
            logger.warning(f"No chunks produced for {file_path.name}")
            return IndexingResult(
                filename=file_path.name,
                document_id=doc_id,
                total_chunks=0,
                text_chunks=0,
                table_chunks=0,
                image_chunks=0,
                error="no_chunks_produced",
            )

        # --- Embed ---
        logger.info(f"Embedding {len(chunks)} chunks for: {file_path.name}")
        texts = [c.get_embedding_text() for c in chunks]
        embeddings = self.embeddings.embed_batch(texts)

        # --- Store ---
        doc_meta = {
            "filename": parsed.metadata.filename,
            "title": parsed.metadata.title or file_path.stem,
            "file_hash": parsed.metadata.file_hash,
        }
        self.vectordb.add_chunks(chunks, embeddings, doc_meta)

        # --- Persist parsed JSON (optional, useful for debugging) ---
        if self.save_parsed:
            out = self.processed_dir / f"{file_path.stem}.json"
            out.write_text(parsed.model_dump_json(indent=2))

        elapsed = time.time() - start
        logger.info(f"Indexed '{file_path.name}' in {elapsed:.1f}s — {len(chunks)} chunks")

        return IndexingResult(
            filename=file_path.name,
            document_id=doc_id,
            total_chunks=result.total_chunks,
            text_chunks=result.text_chunks,
            table_chunks=result.table_chunks,
            image_chunks=result.image_chunks,
            processing_time_seconds=elapsed,
        )

    def index_batch(
        self,
        file_paths: list[str | Path],
        force_reindex: bool = False,
    ) -> list[IndexingResult]:
        """Index multiple PDF documents sequentially."""
        results = []
        for i, path in enumerate(file_paths, 1):
            logger.info(f"[{i}/{len(file_paths)}] {Path(path).name}")
            try:
                results.append(self.index_document(path, force_reindex=force_reindex))
            except Exception as e:
                logger.error(f"Failed to index {path}: {e}")
                results.append(IndexingResult(
                    filename=Path(path).name,
                    document_id="",
                    total_chunks=0,
                    text_chunks=0,
                    table_chunks=0,
                    image_chunks=0,
                    error=str(e),
                ))
        return results
