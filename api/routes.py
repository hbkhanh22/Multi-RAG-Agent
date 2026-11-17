"""
API Routes - REST API endpoints for multi-RAG agent system

Handlers are plain `def` (not `async def`): the workflow is synchronous and can
take many seconds, so FastAPI runs them in its thread pool instead of blocking
the event loop.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.graph import run_workflow, stream_workflow
from src.graph.nodes import router_agent
from src.memory import conversation_store
from src.retrieval import vector_store_manager
from api.schemas import (
    QueryRequest,
    QueryResponse,
    BatchQueryRequest,
    BatchQueryResponse,
    CitationModel,
    HealthResponse,
    StatsResponse,
)
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Agent"])

DOMAINS = ["finance", "tech", "general"]


def _internal_error(context: str, exc: Exception) -> HTTPException:
    """Log full details server-side, return only a generic message + trace id."""
    trace_id = uuid4().hex[:8]
    logger.error(f"[{trace_id}] {context}: {exc}", exc_info=True)
    return HTTPException(status_code=500, detail=f"Internal server error (trace id: {trace_id})")


def _open_conversation(conversation_id: Optional[str]) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """Resolve the conversation (creating one if needed) and load its recent history."""
    try:
        if conversation_id and conversation_store.exists(conversation_id):
            return conversation_id, conversation_store.get_recent(conversation_id, settings.memory_max_turns * 2)
        return conversation_store.create_conversation(), []
    except Exception as e:  # memory must never take chat down
        logger.error(f"Conversation store unavailable: {e}", exc_info=True)
        return None, []


def _remember(cid: Optional[str], query: str, resp: QueryResponse) -> None:
    """Persist a finished turn; failed answers are not stored (they would pollute context)."""
    if not cid or not resp.success or not resp.final_answer:
        return
    try:
        conversation_store.add_exchange(cid, query, resp.final_answer, meta={
            "route": resp.route,
            "confidence": resp.confidence,
            "success": resp.success,
            "sources": [s.model_dump(exclude_none=True) for s in resp.sources],
            "key_themes": resp.key_themes,
            "processing_time": (resp.metadata or {}).get("processing_time"),
        })
    except Exception as e:
        logger.error(f"Failed to save conversation turn: {e}", exc_info=True)


def _build_response(query: str, result: Dict[str, Any], include_metadata: bool = False) -> QueryResponse:
    success = result.get("status") == "complete"
    return QueryResponse(
        query=query,
        success=success,
        final_answer=result.get("final_answer", ""),
        route=result.get("route", "general"),
        confidence=result.get("router_confidence", 0.0),
        key_themes=result.get("key_themes", []),
        sources=[CitationModel(**src) for src in result.get("sources", []) if isinstance(src, dict)],
        agent_responses=result.get("agent_responses", {}),
        metadata={
            "processing_time": result.get("processing_times", {}).get("total", 0),
            "status": result.get("status"),
            "retrieval_stats": result.get("retrieval_stats", {}),
        } if include_metadata else None,
    )


@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """Process a query through the multi-RAG agent system"""
    try:
        logger.info(f"Processing query: {request.query}")
        cid, history = _open_conversation(request.conversation_id)
        result = run_workflow(request.query, debug=request.debug, history=history)
        resp = _build_response(request.query, result, request.include_metadata)
        resp.conversation_id = cid
        _remember(cid, request.query, resp)
        return resp
    except Exception as e:
        raise _internal_error("query failed", e)


def _sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


@router.post("/query/stream")
def query_stream_endpoint(request: QueryRequest):
    """
    Same as /query but streamed as Server-Sent Events:
    `stage` / `route` / `token` events while running, then one `final` event
    carrying the full QueryResponse (sources, themes...), or an `error` event.
    """
    def events():
        try:
            cid, history = _open_conversation(request.conversation_id)
            for event in stream_workflow(request.query, history):
                if event.get("type") == "final":
                    resp = _build_response(request.query, event["state"], request.include_metadata)
                    resp.conversation_id = cid
                    _remember(cid, request.query, resp)
                    yield _sse({"type": "final", "response": resp.model_dump()})
                elif event.get("type") == "error":
                    trace_id = uuid4().hex[:8]
                    logger.error(f"[{trace_id}] stream failed: {event.get('message')}")
                    yield _sse({"type": "error", "message": f"Internal server error (trace id: {trace_id})"})
                else:
                    yield _sse(event)
        except Exception as e:
            trace_id = uuid4().hex[:8]
            logger.error(f"[{trace_id}] stream failed: {e}", exc_info=True)
            yield _sse({"type": "error", "message": f"Internal server error (trace id: {trace_id})"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/query/batch", response_model=BatchQueryResponse)
def batch_query_endpoint(request: BatchQueryRequest):
    """Process multiple queries (sequentially)"""
    if len(request.queries) > settings.max_batch_queries:
        raise HTTPException(
            status_code=422,
            detail=f"At most {settings.max_batch_queries} queries per batch request",
        )
    try:
        logger.info(f"Processing {len(request.queries)} queries")
        results = [_build_response(q, run_workflow(q)) for q in request.queries]
        return BatchQueryResponse(
            results=results,
            total_count=len(results),
            success_count=sum(1 for r in results if r.success),
        )
    except Exception as e:
        raise _internal_error("batch failed", e)


@router.get("/conversations")
def list_conversations_endpoint(limit: int = 50):
    """Recent conversations (newest first) for the sidebar"""
    try:
        return {"conversations": conversation_store.list_conversations(min(max(limit, 1), 200))}
    except Exception as e:
        raise _internal_error("list conversations failed", e)


@router.get("/conversations/{conversation_id}")
def get_conversation_endpoint(conversation_id: str):
    """Full message history of one conversation (to restore a chat)"""
    try:
        if not conversation_store.exists(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"id": conversation_id, "messages": conversation_store.get_messages(conversation_id)}
    except HTTPException:
        raise
    except Exception as e:
        raise _internal_error("get conversation failed", e)


@router.delete("/conversations/{conversation_id}")
def delete_conversation_endpoint(conversation_id: str):
    try:
        if not conversation_store.delete_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"deleted": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        raise _internal_error("delete conversation failed", e)


@router.post("/route")
def route_endpoint(request: QueryRequest):
    """Get routing decision for a query without processing it"""
    try:
        result = router_agent.run({"query": request.query})
        return {
            "query": request.query,
            "route": result.get("route"),
            "agents": result.get("agents"),
            "confidence": result.get("confidence"),
            "reasoning": result.get("reasoning"),
        }
    except Exception as e:
        raise _internal_error("routing failed", e)


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check: verifies vector stores and LLM credentials are usable"""
    components: Dict[str, str] = {}
    for domain in DOMAINS:
        stats = vector_store_manager.get_store_stats(domain)
        components[f"vector_store_{domain}"] = "healthy" if "error" not in stats else "unavailable"
    components["llm_api_key"] = "configured" if settings.nvidia_api_key else "missing"

    degraded = any(v in ("unavailable", "missing") for v in components.values())
    return HealthResponse(
        status="degraded" if degraded else "healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        components=components,
    )


@router.get("/stats", response_model=StatsResponse)
def stats_endpoint():
    """Get system statistics"""
    try:
        domains = {d: vector_store_manager.get_store_stats(d).get("document_count", 0) for d in DOMAINS}
        return StatsResponse(
            domains=domains,
            agents_available=DOMAINS + ["web", "supervisor"],
            uptime=0.0,
        )
    except Exception as e:
        raise _internal_error("stats failed", e)


@router.get("/domain/{domain}/stats")
def domain_stats_endpoint(domain: str):
    """Get statistics for a specific domain"""
    if domain not in DOMAINS:
        raise HTTPException(status_code=400, detail="Invalid domain")
    try:
        return vector_store_manager.get_store_stats(domain)
    except Exception as e:
        raise _internal_error("domain stats failed", e)


@router.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest):
    """Chat endpoint (alias of /query, used by the frontend)"""
    return query_endpoint(request)
