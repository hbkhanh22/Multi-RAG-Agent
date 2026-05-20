"""
Workflow - LangGraph workflow builder for multi-RAG agent system
"""

import logging
from typing import Any, Dict
import time

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from .state import AgentState
from .nodes import (
    router_node,
    finance_rag_node,
    tech_rag_node,
    general_rag_node,
    supervisor_node,
    synthesizer_node
)
from .edges import route_after_router, route_to_end

logger = logging.getLogger(__name__)


def build_agent_graph():
    """Build LangGraph workflow for multi-RAG agent system"""
    
    graph = StateGraph(AgentState)
    
    # Add nodes
    logger.info("Adding nodes to graph...")
    graph.add_node("router", router_node)
    graph.add_node("finance_agent", finance_rag_node)
    graph.add_node("tech_agent", tech_rag_node)
    graph.add_node("general_agent", general_rag_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("synthesizer", synthesizer_node)
    
    # Set entry point
    graph.set_entry_point("router")
    
    # Add conditional edges from router
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "supervisor": "supervisor",
            "finance_agent": "finance_agent",
            "tech_agent": "tech_agent",
            "general_agent": "general_agent",
        }
    )
    
    # Route from agents to synthesizer
    graph.add_edge("finance_agent", "synthesizer")
    graph.add_edge("tech_agent", "synthesizer")
    graph.add_edge("general_agent", "synthesizer")
    graph.add_edge("supervisor", "synthesizer")
    
    # Final edge to END
    graph.add_edge("synthesizer", END)
    
    logger.info("Graph construction complete")
    
    return graph.compile()


def create_initial_state(query: str) -> Dict[str, Any]:
    """Create initial state for graph execution"""
    return {
        "query": query,
        "messages": [HumanMessage(content=query)],
        "route": "general",
        "active_agents": ["general"],
        "router_confidence": 0.0,
        "retrieved_docs": {},
        "reranked_docs": {},
        "agent_responses": {},
        "final_answer": "",
        "key_themes": [],
        "sources": [],
        "iteration": 0,
        "max_iterations": 1,
        "error": "",
        "status": "pending",
        "start_time": None,
        "processing_times": {}
    }


def run_workflow(query: str, debug: bool = False) -> Dict[str, Any]:
    """
    Run the multi-RAG agent workflow
    
    Args:
        query: User query
        debug: Enable debug logging
    
    Returns:
        Final state with answer and metadata
    """
    try:
        # Build workflow
        workflow = build_agent_graph()
        
        # Create initial state
        initial_state = create_initial_state(query)
        start_time = time.time()
        initial_state["start_time"] = start_time
        
        if debug:
            logger.setLevel(logging.DEBUG)
            logger.info(f"Starting workflow with query: {query}")
        
        # Execute workflow
        final_state = workflow.invoke(initial_state)
        
        # Calculate processing time
        duration = time.time() - start_time
        final_state["processing_times"]["total"] = duration
        logger.info(f"Workflow completed in {duration:.2f} seconds")
        
        final_state["status"] = "complete"
        
        return final_state
    
    except Exception as e:
        logger.error(f"Error running workflow: {e}")
        return {
            "status": "error",
            "error": str(e),
            "final_answer": f"Error processing query: {str(e)}",
            "sources": [],
            "query": query
        }


# Compile graph as singleton
try:
    app = build_agent_graph()
    logger.info("Agent graph compiled successfully")
except Exception as e:
    logger.error(f"Failed to compile agent graph: {e}")
    app = None