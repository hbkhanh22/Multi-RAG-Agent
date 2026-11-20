"""
Document management API: upload / list / delete documents and trigger re-indexing.

Uploads are stored (local data/raw or S3), registered as "pending" and indexed by the
ingestion worker in the background; clients poll GET /api/documents or the job.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.deps import require_admin
from config.settings import settings
from src.ingestion.auto_ingest import get_document, list_documents, mark_pending
from src.storage import jobs
from src.storage.objects import DOMAINS, get_storage, safe_filename

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Documents"], dependencies=[Depends(require_admin)])


@router.get("/documents")
def documents_endpoint(domain: Optional[str] = None):
    """Registered documents of the active index with their indexing status."""
    docs = list_documents()
    if domain:
        docs = [d for d in docs if d["domain"] == domain]
    counts = {}
    for d in docs:
        counts[d["status"]] = counts.get(d["status"], 0) + 1
    return {"documents": docs, "total": len(docs), "by_status": counts,
            "storage": settings.document_storage}


@router.post("/documents", status_code=202)
def upload_document(file: UploadFile = File(...), domain: str = Form("general")):
    """Store a file and queue its indexing (202 Accepted + job id)."""
    if domain not in DOMAINS:
        raise HTTPException(status_code=422, detail=f"domain must be one of {list(DOMAINS)}")
    try:
        filename = safe_filename(file.filename or "")
    except ValueError:
        raise HTTPException(status_code=415,
                            detail=f"Unsupported file type; allowed: {settings.supported_file_types}")

    # size check without trusting Content-Length
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size == 0:
        raise HTTPException(status_code=422, detail="Empty file")
    if size > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File larger than {settings.max_upload_mb} MB")

    key = get_storage().put(domain, filename, file.file)
    doc_id = mark_pending(domain, key, filename, size)
    job_id = jobs.enqueue("sync", {"reason": "upload"})
    logger.info(f"Uploaded {filename} ({size} bytes) to {domain} as {key}; job #{job_id}")
    return {"document_id": doc_id, "job_id": job_id, "key": key, "status": "pending"}


@router.delete("/documents/{doc_id}", status_code=202)
def delete_document(doc_id: int):
    """Delete the stored file; the worker then removes its chunks from the index."""
    doc = get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    job_id = jobs.enqueue("delete", {"key": doc["source_key"]}, dedupe=True)
    return {"document_id": doc_id, "job_id": job_id, "status": "deleting"}


@router.post("/ingest/sync", status_code=202)
def sync_endpoint():
    """Re-scan document storage and index whatever changed."""
    return {"job_id": jobs.enqueue("sync", {"reason": "manual"})}


@router.get("/jobs")
def jobs_endpoint(limit: int = 20):
    return {"jobs": jobs.recent(min(max(limit, 1), 100))}


@router.get("/jobs/{job_id}")
def job_endpoint(job_id: int):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
