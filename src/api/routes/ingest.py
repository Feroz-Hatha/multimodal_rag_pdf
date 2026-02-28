"""PDF upload and indexing endpoint — returns a job ID for progress polling."""

import logging
import threading
import uuid
from datetime import datetime

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from src.api.models import IngestStartResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _run_indexing(app_state, job_id: str, upload_path, filename: str) -> None:
    """
    Background thread that runs the full indexing pipeline and updates
    the shared jobs dict with progress information at each stage.
    """
    jobs = app_state.jobs
    pipeline = app_state.pipeline
    registry = app_state.registry

    def update(progress: float, stage: str, **extra):
        jobs[job_id].update({"progress": progress, "stage": stage, **extra})

    try:
        update(0.05, "Starting…")

        # --- Parse ---
        update(0.10, "Parsing PDF…")
        parsed = pipeline.parser.parse(upload_path)
        doc_id = str(parsed.metadata.document_id)

        # --- Deduplication ---
        existing_id = pipeline.vectordb.file_hash_exists(parsed.metadata.file_hash)
        if existing_id:
            # Already in ChromaDB — resolve chunk counts from registry if available
            reg_entry = registry.get(existing_id)
            if reg_entry:
                total = reg_entry.get("total_chunks", 0)
                text_c = reg_entry.get("text_chunks", 0)
                table_c = reg_entry.get("table_chunks", 0)
                image_c = reg_entry.get("image_chunks", 0)
            else:
                total = text_c = table_c = image_c = 0
            jobs[job_id].update({
                "status": "done",
                "progress": 1.0,
                "stage": "Already indexed",
                "document_id": existing_id,
                "filename": filename,
                "total_chunks": total,
                "text_chunks": text_c,
                "table_chunks": table_c,
                "image_chunks": image_c,
                "already_indexed": True,
            })
            return

        # --- Image descriptions ---
        if pipeline.image_describer:
            n_images = sum(
                1 for item in parsed.content_items
                if item.get("type") == "image" and item.get("image_data")
            )
            if n_images:
                update(0.25, f"Describing {n_images} image(s) with Claude Vision…")
                pipeline.image_describer.describe_document_images(parsed.content_items)

        # Strip any remaining image_data before JSON serialisation
        for item in parsed.content_items:
            item.pop("image_data", None)

        # --- Chunk ---
        update(0.55, "Chunking document…")
        result = pipeline.chunker.chunk(parsed)
        chunks = result.chunks

        if not chunks:
            logger.warning(f"Job {job_id}: no chunks produced for '{filename}' — document may be too short or have unrecognised structure")
            jobs[job_id].update({
                "status": "error",
                "progress": 1.0,
                "stage": "No chunks produced",
                "error": "no_chunks_produced",
            })
            return

        # --- Embed ---
        update(0.70, f"Embedding {len(chunks)} chunks…")
        texts = [c.get_embedding_text() for c in chunks]
        embeddings = pipeline.embeddings.embed_batch(texts)

        # --- Store ---
        update(0.90, "Storing in vector database…")
        doc_meta = {
            "filename": parsed.metadata.filename,
            "title": parsed.metadata.title or upload_path.stem,
            "file_hash": parsed.metadata.file_hash,
        }
        pipeline.vectordb.add_chunks(chunks, embeddings, doc_meta)

        # --- Persist parsed JSON ---
        if pipeline.save_parsed:
            out = pipeline.processed_dir / f"{upload_path.stem}.json"
            out.write_text(parsed.model_dump_json(indent=2))

        # --- Update registry ---
        registry.add({
            "document_id": doc_id,
            "filename": filename,
            "title": filename,
            "total_chunks": result.total_chunks,
            "text_chunks": result.text_chunks,
            "table_chunks": result.table_chunks,
            "image_chunks": result.image_chunks,
            "indexed_at": datetime.utcnow().isoformat(),
        })

        jobs[job_id].update({
            "status": "done",
            "progress": 1.0,
            "stage": "Complete",
            "document_id": doc_id,
            "filename": filename,
            "total_chunks": result.total_chunks,
            "text_chunks": result.text_chunks,
            "table_chunks": result.table_chunks,
            "image_chunks": result.image_chunks,
            "already_indexed": False,
        })
        logger.info(f"Job {job_id}: indexed '{filename}' — {result.total_chunks} chunks")

    except Exception as exc:
        logger.exception(f"Job {job_id}: indexing failed for '{filename}'")
        jobs[job_id].update({
            "status": "error",
            "progress": 1.0,
            "stage": "Failed",
            "error": str(exc),
        })


@router.post("/ingest", response_model=IngestStartResponse)
def ingest_document(request: Request, file: UploadFile = File(...)):
    """
    Upload a PDF and start background indexing.

    Returns a job_id immediately. Poll GET /api/v1/jobs/{job_id} for progress.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file to disk
    upload_dir = request.app.state.settings.upload_dir
    upload_path = upload_dir / file.filename
    upload_path.write_bytes(file.file.read())
    logger.info(f"Saved upload: {upload_path}")

    # Create job record
    job_id = str(uuid.uuid4())
    request.app.state.jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "stage": "Queued",
        "filename": file.filename,
        "document_id": None,
        "total_chunks": None,
        "text_chunks": None,
        "table_chunks": None,
        "image_chunks": None,
        "already_indexed": False,
        "error": None,
    }

    # Start background thread
    thread = threading.Thread(
        target=_run_indexing,
        args=(request.app.state, job_id, upload_path, file.filename),
        daemon=True,
    )
    thread.start()

    return IngestStartResponse(job_id=job_id, filename=file.filename)
