"""
Graph Edges - Routing logic between nodes
"""

from typing import Literal
import logging

from .state import AgentState

logger = logging.getLogger(__name__)


def route_after_router(state: AgentState) -> Literal["supervisor", "finance_agent", "tech_agent", "general_agent", "synthesizer"]:
    """
    Route after router node
    
    Determines next node based on routing decision
    """
    route = state.get("route", "general")
    active_agents = state.get("active_agents", ["general"])
    
    logger.info(f"Routing decision: {route}, agents: {active_agents}")
    
    # If multi-domain query, go to supervisor
    if route == "multi" or len(active_agents) > 1:
        logger.info("Multi-domain query -> Supervisor")
        return "supervisor"
    
    # Single domain routing
    domain = active_agents[0] if active_agents else "general"
    
    if domain == "finance":
        return "finance_agent"
    elif domain == "tech":
        return "tech_agent"
    else:
        return "general_agent"


def route_to_synthesizer(state: AgentState) -> Literal["synthesizer"]:
    """Always route to synthesizer after agent processing"""
    return "synthesizer"


def route_to_end(state: AgentState) -> Literal["__end__"]:
    """Route to end"""
    return "__end__"


def should_continue(state: AgentState) -> Literal["router", "__end__"]:
    """Determine if we should continue or end"""
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 1)
    
    if iteration >= max_iterations:
        logger.info("Max iterations reached, ending")
        return "__end__"
    
    return "router"
