"""Document listing and deletion endpoints."""

from fastapi import APIRouter, HTTPException, Request

from src.api.models import DeleteResponse, DocumentInfo

router = APIRouter()


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents(request: Request):
    """Return all indexed documents from the registry."""
    return request.app.state.registry.list()


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str, request: Request):
    """
    Remove a document and all its chunks from ChromaDB and the registry.
    """
    if not request.app.state.registry.get(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")

    deleted = request.app.state.vectordb.delete_document(document_id)
    request.app.state.registry.remove(document_id)

    return DeleteResponse(document_id=document_id, deleted_chunks=deleted)
