"""Pydantic models for API request and response payloads."""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    n_results: int = 5
    document_id: str | None = None
    document_ids: list[str] | None = None
    content_type: str | None = None


class SourceInfo(BaseModel):
    filename: str
    heading: str
    section_hierarchy: list[str]
    page_numbers: list[int]
    content_type: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceInfo]
    model_id: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    title: str
    total_chunks: int
    text_chunks: int
    table_chunks: int
    image_chunks: int
    indexed_at: str


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    total_chunks: int
    text_chunks: int
    table_chunks: int
    image_chunks: int
    processing_time_seconds: float
    message: str  # "indexed" | "already_indexed"


class DeleteResponse(BaseModel):
    document_id: str
    deleted_chunks: int


class HealthResponse(BaseModel):
    status: str
    total_chunks: int
    total_documents: int


class IngestStartResponse(BaseModel):
    job_id: str
    filename: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending" | "running" | "done" | "error"
    progress: float  # 0.0 – 1.0
    stage: str
    filename: str
    # Populated on completion
    document_id: str | None = None
    total_chunks: int | None = None
    text_chunks: int | None = None
    table_chunks: int | None = None
    image_chunks: int | None = None
    already_indexed: bool = False
    error: str | None = None
