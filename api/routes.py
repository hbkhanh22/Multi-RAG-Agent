"""
API Routes - REST API endpoints for multi-RAG agent system
"""

from typing import Optional
import logging
from datetime import datetime

from fastapi import APIRouter, Query, BackgroundTasks, HTTPException
from src.graph import run_workflow
from src.retrieval import vector_store_manager
from src.agents import RouterAgent
from api.schemas import (
    QueryRequest,
    QueryResponse,
    BatchQueryRequest,
    BatchQueryResponse,
    CitationModel,
    HealthResponse,
    StatsResponse
)
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Agent"])


# Initialize components
router_agent = RouterAgent()


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Process a query through the multi-RAG agent system
    
    Args:
        request: Query request with optional domain filter
    
    Returns:
        QueryResponse with answer and citations
    """
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Run workflow
        result = run_workflow(request.query, debug=request.debug)
        
        # Build response
        response = QueryResponse(
            query=request.query,
            success=result.get("status") == "complete",
            final_answer=result.get("final_answer", ""),
            route=result.get("route", "general"),
            confidence=result.get("router_confidence", 0.0),
            key_themes=result.get("key_themes", []),
            sources=[
                CitationModel(**src) if isinstance(src, dict) else src
                for src in result.get("sources", [])
            ],
            agent_responses=result.get("agent_responses", {}),
            metadata={
                "processing_time": result.get("processing_times", {}).get("total", 0),
                "status": result.get("status")
            } if request.include_metadata else None
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/batch", response_model=BatchQueryResponse)
async def batch_query_endpoint(request: BatchQueryRequest):
    """
    Process multiple queries
    
    Args:
        request: Batch query request
    
    Returns:
        BatchQueryResponse with results
    """
    try:
        logger.info(f"Processing {len(request.queries)} queries")
        
        results = []
        for query_str in request.queries:
            result = run_workflow(query_str)
            
            response = QueryResponse(
                query=query_str,
                success=result.get("status") == "complete",
                final_answer=result.get("final_answer", ""),
                route=result.get("route", "general"),
                confidence=result.get("router_confidence", 0.0),
                key_themes=result.get("key_themes", []),
                sources=result.get("sources", []),
                agent_responses=result.get("agent_responses", {})
            )
            results.append(response)
        
        success_count = sum(1 for r in results if r.success)
        
        return BatchQueryResponse(
            results=results,
            total_count=len(results),
            success_count=success_count
        )
    
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/route")
async def route_endpoint(request: QueryRequest):
    """
    Get routing decision for a query without processing
    
    Returns routing decision and confidence score
    """
    try:
        result = router_agent.run({"query": request.query})
        
        return {
            "query": request.query,
            "route": result.get("route"),
            "agents": result.get("agents"),
            "confidence": result.get("confidence"),
            "reasoning": result.get("reasoning")
        }
    
    except Exception as e:
        logger.error(f"Error in routing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        components = {
            "router_agent": "healthy",
            "rag_agents": "healthy",
            "vector_stores": "healthy"
        }
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
            components=components
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.get("/stats", response_model=StatsResponse)
async def stats_endpoint():
    """Get system statistics"""
    try:
        domains = {}
        for domain in ["finance", "tech", "general"]:
            stats = vector_store_manager.get_store_stats(domain)
            domains[domain] = stats.get("document_count", 0)
        
        return StatsResponse(
            domains=domains,
            agents_available=["finance", "tech", "general", "supervisor"],
            uptime=0.0  # Could track actual uptime
        )
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/domain/{domain}/stats")
async def domain_stats_endpoint(domain: str):
    """Get statistics for a specific domain"""
    try:
        if domain not in ["finance", "tech", "general"]:
            raise HTTPException(status_code=400, detail="Invalid domain")
        
        stats = vector_store_manager.get_store_stats(domain)
        return stats
    
    except Exception as e:
        logger.error(f"Error getting domain stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat_endpoint(request: QueryRequest):
    """
    Chat endpoint for conversational queries
    (Same as /query but with different naming)
    """
    return await query_endpoint(request)
