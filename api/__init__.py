"""
API module - FastAPI application for Multi-RAG Agent System
"""

from api.main import app
from api.routes import router
from api.schemas import (
    QueryRequest,
    QueryResponse,
    BatchQueryRequest,
    BatchQueryResponse
)

__all__ = [
    "app",
    "router",
    "QueryRequest",
    "QueryResponse",
    "BatchQueryRequest",
    "BatchQueryResponse"
]
