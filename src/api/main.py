"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import HealthResponse, JobStatusResponse
from src.api.registry import DocumentRegistry
from src.api.routes import documents, ingest, query
from src.config import settings
from src.embeddings.bedrock_embeddings import BedrockEmbeddings
from src.generation.image_describer import ImageDescriber
from src.generation.rag_pipeline import RAGPipeline
from src.generation.response_generator import ResponseGenerator
from src.ingestion.indexing_pipeline import IndexingPipeline
from src.retrieval.retriever import Retriever
from src.vectordb.chroma_client import ChromaVectorDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise and tear down shared components."""
    logger.info("Starting up — initialising components...")
    settings.ensure_directories()

    # Shared AWS credential kwargs
    aws_kwargs = dict(
        region=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )

    embeddings = BedrockEmbeddings(
        model_id=settings.bedrock_embedding_model_id,
        **aws_kwargs,
    )
    vectordb = ChromaVectorDB(persist_dir=settings.chroma_db_dir)
    image_describer = ImageDescriber(
        model_id=settings.bedrock_llm_model_id,
        **aws_kwargs,
    )
    generator = ResponseGenerator(
        model_id=settings.bedrock_llm_model_id,
        **aws_kwargs,
    )

    registry = DocumentRegistry(path=settings.processed_dir / "registry.json")
    registry.seed_from_chroma(vectordb)  # surface CLI-indexed docs in the UI

    app.state.settings = settings
    app.state.embeddings = embeddings
    app.state.vectordb = vectordb
    app.state.pipeline = IndexingPipeline(
        embeddings=embeddings,
        vectordb=vectordb,
        image_describer=image_describer,
        processed_dir=settings.processed_dir,
    )
    app.state.rag = RAGPipeline(
        embeddings=embeddings,
        vectordb=vectordb,
        generator=generator,
    )
    app.state.registry = registry
    app.state.jobs = {}  # job_id → job status dict (populated by ingest background threads)

    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="PDF RAG API",
    description="Retrieval-Augmented Generation over uploaded PDF documents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
def health():
    stats = app.state.vectordb.get_stats()
    return HealthResponse(
        status="ok",
        total_chunks=stats["total_chunks"],
        total_documents=len(app.state.registry.list()),
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse, tags=["Ingest"])
def get_job_status(job_id: str):
    """Poll indexing job progress. Status: pending → running → done | error."""
    from fastapi import HTTPException as _HTTPException
    job = app.state.jobs.get(job_id)
    if job is None:
        raise _HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    return JobStatusResponse(job_id=job_id, **job)
