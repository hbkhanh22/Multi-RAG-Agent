"""
API Schemas - Pydantic models for request/response validation
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# Request Models
class QueryRequest(BaseModel):
    """Query request"""
    query: str = Field(..., min_length=1, max_length=2000)
    domain: Optional[str] = Field(None, description="Optional domain filter")
    include_metadata: bool = Field(default=False)
    debug: bool = Field(default=False)


class BatchQueryRequest(BaseModel):
    """Batch query request"""
    queries: List[str] = Field(..., min_items=1, max_items=100)
    domain: Optional[str] = None


# Response Models
class CitationModel(BaseModel):
    """Citation in response"""
    agent: str
    source_id: Optional[int] = None
    source: str
    page: Optional[int] = None


class AgentResponseModel(BaseModel):
    """Single agent response"""
    agent: str
    success: bool
    answer: str
    retrieved_count: int
    reranked_count: int
    sources: List[CitationModel] = []


class QueryResponse(BaseModel):
    """Query response"""
    query: str
    success: bool
    final_answer: str
    route: str
    confidence: float = 0.0
    key_themes: List[str] = []
    sources: List[CitationModel] = []
    agent_responses: Dict[str, Any] = {}
    metadata: Optional[Dict] = None


class BatchQueryResponse(BaseModel):
    """Batch query response"""
    results: List[QueryResponse]
    total_count: int
    success_count: int


# Status Models
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    components: Dict[str, str] = {}


class StatsResponse(BaseModel):
    """System statistics"""
    domains: Dict[str, int]  # {domain: document_count}
    agents_available: List[str]
    memory_usage: Optional[Dict] = None
    uptime: float


# Error Model
class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    message: str
    code: int
    timestamp: str
    trace_id: Optional[str] = None
