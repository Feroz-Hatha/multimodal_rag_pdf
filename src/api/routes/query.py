"""RAG query endpoint."""

import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.api.models import QueryRequest, QueryResponse, SourceInfo
from src.generation.rag_pipeline import RAGResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request):
    """
    Run a RAG query: embed the question, retrieve relevant chunks,
    generate a grounded answer with inline citations.
    """
    response = request.app.state.rag.query(
        question=body.question,
        n_results=body.n_results,
        document_id=body.document_id,
        document_ids=body.document_ids,
        content_type=body.content_type,
    )

    return QueryResponse(
        question=response.question,
        answer=response.answer,
        sources=[
            SourceInfo(
                filename=s.filename,
                heading=s.heading,
                section_hierarchy=s.section_hierarchy,
                page_numbers=s.page_numbers,
                content_type=s.content_type,
                score=s.score,
            )
            for s in response.sources
        ],
        model_id=response.model_id,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        estimated_cost_usd=response.estimated_cost_usd(),
    )


@router.post("/query/stream")
def query_stream(body: QueryRequest, request: Request):
    """
    Streaming variant of /query. Returns an SSE stream.

    Each event is a JSON object:
      {"type": "delta",  "text": "..."}          — partial answer token(s)
      {"type": "done",   "sources": [...]}        — final event with source list
      {"type": "error",  "message": "..."}        — if something goes wrong
    """
    def generate():
        try:
            for item in request.app.state.rag.query_stream(
                question=body.question,
                n_results=body.n_results,
                document_id=body.document_id,
                document_ids=body.document_ids,
                content_type=body.content_type,
            ):
                if isinstance(item, RAGResponse):
                    sources = [
                        {
                            "filename": s.filename,
                            "heading": s.heading,
                            "section_hierarchy": s.section_hierarchy,
                            "page_numbers": s.page_numbers,
                            "content_type": s.content_type,
                            "score": s.score,
                        }
                        for s in item.sources
                    ]
                    yield f"data: {json.dumps({'type': 'done', 'sources': sources})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'delta', 'text': item})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
