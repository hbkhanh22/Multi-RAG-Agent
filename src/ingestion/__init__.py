"""
Ingestion module - chunking, optional page context and synchronisation of the
document storage with the vector index
"""

from .chunker import DocumentChunker
from .auto_ingest import auto_ingest_raw_documents, sync_documents

__all__ = [
    "DocumentChunker",
    "auto_ingest_raw_documents",
    "sync_documents",
]
