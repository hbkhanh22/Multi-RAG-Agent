"""
Workflow - LangGraph workflow builder for multi-RAG agent system
"""

import logging
import time
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from .state import AgentState
from .nodes import (
    condense_node,
    router_node,
    finance_rag_node,
    tech_rag_node,
    general_rag_node,
    web_agent_node,
    supervisor_node,
    synthesizer_node,
    direct_chat_node,
)
from .edges import route_after_router

logger = logging.getLogger(__name__)


def build_agent_graph():
    """Build and compile the LangGraph workflow for the multi-RAG agent system"""
    graph = StateGraph(AgentState)

    graph.add_node("condense", condense_node)
    graph.add_node("router", router_node)
    graph.add_node("finance_agent", finance_rag_node)
    graph.add_node("tech_agent", tech_rag_node)
    graph.add_node("general_agent", general_rag_node)
    graph.add_node("web_agent", web_agent_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("direct_chat_agent", direct_chat_node)

    graph.set_entry_point("condense")
    graph.add_edge("condense", "router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "supervisor": "supervisor",
            "finance_agent": "finance_agent",
            "tech_agent": "tech_agent",
            "general_agent": "general_agent",
            "web_agent": "web_agent",
            "direct_chat_agent": "direct_chat_agent",
        },
    )

    for node in ("finance_agent", "tech_agent", "general_agent", "web_agent", "supervisor"):
        graph.add_edge(node, "synthesizer")

    graph.add_edge("direct_chat_agent", END)
    graph.add_edge("synthesizer", END)

    return graph.compile()


def create_initial_state(query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Create initial state for graph execution"""
    return {
        "query": query,
        "history": history or [],
        "standalone_query": "",
        "messages": [HumanMessage(content=query)],
        "route": "general",
        "active_agents": ["general"],
        "router_confidence": 0.0,
        "retrieved_docs": {},
        "reranked_docs": {},
        "agent_responses": {},
        "retrieval_stats": {},
        "final_answer": "",
        "key_themes": [],
        "sources": [],
        "iteration": 0,
        "max_iterations": 1,
        "error": "",
        "status": "pending",
        "start_time": None,
        "processing_times": {},
    }


# Compile the graph once (compiling per request is wasteful)
try:
    app = build_agent_graph()
    logger.info("Agent graph compiled successfully")
except Exception as e:  # pragma: no cover
    logger.error(f"Failed to compile agent graph: {e}")
    app = None


def stream_workflow(query: str, history: Optional[List[Dict[str, str]]] = None):
    """
    Run the workflow and yield events as it progresses:
      {"type": "stage"|"route"|"token", ...}  while running
      {"type": "final", "state": {...}}       once, at the end (or {"type": "error"})
    """
    global app
    try:
        if app is None:
            app = build_agent_graph()
        initial_state = create_initial_state(query, history)
        start = time.time()
        initial_state["start_time"] = start

        final_state: Dict[str, Any] = dict(initial_state)
        for mode, payload in app.stream(initial_state, stream_mode=["values", "custom"]):
            if mode == "custom":
                yield payload
            else:
                final_state = payload

        final_state.setdefault("processing_times", {})["total"] = time.time() - start
        if final_state.get("status") == "pending":
            final_state["status"] = "complete" if final_state.get("final_answer") else "error"
        yield {"type": "final", "state": final_state}
    except Exception as e:
        logger.error(f"Error streaming workflow: {e}", exc_info=True)
        yield {"type": "error", "message": str(e)}


def run_workflow(query: str, debug: bool = False, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Run the multi-RAG agent workflow.

    Returns the final state. `status` is "complete" only if an answer was
    actually produced, otherwise "error".
    """
    global app
    try:
        if app is None:
            app = build_agent_graph()

        initial_state = create_initial_state(query, history)
        start_time = time.time()
        initial_state["start_time"] = start_time

        if debug:
            logger.info(f"Starting workflow with query: {query}")

        final_state = app.invoke(initial_state)

        duration = time.time() - start_time
        final_state["processing_times"]["total"] = duration
        logger.info(f"Workflow completed in {duration:.2f} seconds")

        if final_state.get("status") == "pending":
            final_state["status"] = "complete" if final_state.get("final_answer") else "error"

        return final_state

    except Exception as e:
        logger.error(f"Error running workflow: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "final_answer": "Đã xảy ra lỗi khi xử lý câu hỏi. Vui lòng thử lại.",
            "sources": [],
            "query": query,
        }
