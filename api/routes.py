"""
API Routes - REST API endpoints for multi-RAG agent system

Handlers are plain `def` (not `async def`): the workflow is synchronous and can
take many seconds, so FastAPI runs them in its thread pool instead of blocking
the event loop.
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.deps import rate_limit
from src import runtime
from src.graph import run_workflow, stream_workflow
from src.graph.nodes import router_agent
from src.mcp_client import mcp_manager
from src.memory import conversation_store
from src.observability import metrics, tracing
from src.observability.logging import request_id_var
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

DOMAINS = ["sport", "tech", "general"]


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


def _new_run(conversation_id: Optional[str]) -> Tuple[runtime.RunContext, Dict[str, Any]]:
    """Run context (cancel flag + token usage) and extra graph config (tracing)."""
    rid = request_id_var.get()
    ctx = runtime.RunContext(None if rid == "-" else rid)
    return ctx, tracing.run_config(ctx.request_id, conversation_id)


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
            "tool_calls": (resp.agent_responses.get("tools") or {}).get("tool_calls", []),
            "usage": resp.usage,
            "request_id": resp.request_id,
        }, usage=resp.usage)
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
        # Full context chunks are for the judge only; keep the payload small
        agent_responses={
            name: {k: v for k, v in r.items() if k != "contexts"} if isinstance(r, dict) else r
            for name, r in (result.get("agent_responses") or {}).items()
        },
        evaluation={
            **result["evaluation"],
            "refine_rounds": result.get("iteration", 0),
            "history": [
                {k: h.get(k) for k in ("round", "scores", "overall", "passed")}
                for h in result.get("evaluation_history", [])
            ],
        } if result.get("evaluation") else None,
        usage=result.get("usage"),
        metadata={
            "processing_time": result.get("processing_times", {}).get("total", 0),
            "status": result.get("status"),
            "retrieval_stats": result.get("retrieval_stats", {}),
        } if include_metadata else None,
    )


@router.post("/query", response_model=QueryResponse, dependencies=[Depends(rate_limit)])
def query_endpoint(request: QueryRequest):
    """Process a query through the multi-RAG agent system"""
    try:
        logger.info(f"Processing query ({len(request.query)} chars)")
        cid, history = _open_conversation(request.conversation_id)
        ctx, config = _new_run(cid)
        result = run_workflow(request.query, debug=request.debug, history=history, run_ctx=ctx, config=config)
        metrics.record_query(result)
        resp = _build_response(request.query, result, request.include_metadata)
        resp.conversation_id = cid
        resp.request_id = ctx.request_id
        _remember(cid, request.query, resp)
        return resp
    except Exception as e:
        raise _internal_error("query failed", e)


def _sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


@router.post("/query/stream", dependencies=[Depends(rate_limit)])
def query_stream_endpoint(request: QueryRequest):
    """
    Same as /query but streamed as Server-Sent Events:
    one `start` event (request_id, needed to cancel), `stage` / `route` / `token`
    events while running, then one `final` event carrying the full QueryResponse
    (sources, themes, usage...), or a `cancelled` / `error` event.

    Cancel with POST /api/query/{request_id}/cancel (the "Stop" button) or by closing
    the connection: the graph stops before its next step and a streaming LLM call is
    interrupted, so no more tokens are paid for.
    """
    ctx, config = _new_run(request.conversation_id)
    runtime.register(ctx)
    log_rid = request_id_var.get()

    def events():
        token = request_id_var.set(log_rid)   # the generator runs in pool threads: keep the log id
        started = time.time()
        ttft = None
        finished = False
        metrics.stream_started()
        try:
            cid, history = _open_conversation(request.conversation_id)
            yield _sse({"type": "start", "request_id": ctx.request_id, "conversation_id": cid})
            for event in stream_workflow(request.query, history, run_ctx=ctx, config=config):
                kind = event.get("type")
                if kind == "token" and ttft is None:
                    ttft = time.time() - started
                if kind == "final":
                    metrics.record_query(event["state"], ttft)
                    resp = _build_response(request.query, event["state"], request.include_metadata)
                    resp.conversation_id = cid
                    resp.request_id = ctx.request_id
                    _remember(cid, request.query, resp)
                    yield _sse({"type": "final", "response": resp.model_dump()})
                elif kind == "cancelled":
                    metrics.record_cancelled()
                    yield _sse({"type": "cancelled", "request_id": ctx.request_id, "usage": ctx.usage()})
                elif kind == "error":
                    trace_id = uuid4().hex[:8]
                    logger.error(f"[{trace_id}] stream failed: {event.get('message')}")
                    yield _sse({"type": "error", "message": f"Internal server error (trace id: {trace_id})"})
                else:
                    yield _sse(event)
            finished = True
        except GeneratorExit:      # client went away: stop generating (and paying for) tokens
            ctx.cancel()
            raise
        except Exception as e:
            trace_id = uuid4().hex[:8]
            logger.error(f"[{trace_id}] stream failed: {e}", exc_info=True)
            yield _sse({"type": "error", "message": f"Internal server error (trace id: {trace_id})"})
        finally:
            if not finished:
                ctx.cancel()
            runtime.unregister(ctx)
            metrics.stream_finished()
            try:
                request_id_var.reset(token)
            except ValueError:   # reset from a different thread/context
                pass

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/query/{request_id}/cancel")
def cancel_query_endpoint(request_id: str):
    """Stop an answer that is being generated (the UI "Stop" button)."""
    if not runtime.cancel(request_id):
        raise HTTPException(status_code=404, detail="No running answer with this request id")
    return {"cancelled": request_id}


@router.post("/query/batch", response_model=BatchQueryResponse, dependencies=[Depends(rate_limit)])
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


@router.get("/mcp/status")
def mcp_status_endpoint():
    """Connected MCP servers and the tools the agent may call"""
    status = mcp_manager.status()
    status["tools"] = [
        {"name": spec["function"]["name"], "description": spec["function"]["description"][:200]}
        for spec in mcp_manager.tool_specs()
    ]
    return status


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
    mcp = mcp_manager.status()
    components["mcp"] = (
        "disabled" if not mcp["enabled"]
        else f"{mcp['tool_count']} tools" if mcp["tool_count"]
        else "connecting" if not mcp["ready"] else "no tools"
    )

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


@router.post("/chat", response_model=QueryResponse, dependencies=[Depends(rate_limit)])
def chat_endpoint(request: QueryRequest):
    """Chat endpoint (alias of /query, used by the frontend)"""
    return query_endpoint(request)
