"""
Graph Edges - Routing logic between nodes
"""

from typing import Literal
import logging

from .state import AgentState

logger = logging.getLogger(__name__)

NextNode = Literal[
    "supervisor", "finance_agent", "tech_agent", "general_agent",
    "web_agent", "tool_agent", "direct_chat_agent",
]


def route_after_router(state: AgentState) -> NextNode:
    """Pick the next node from the router's decision."""
    route = state.get("route", "general")
    active_agents = state.get("active_agents", ["general"])

    logger.info(f"Routing decision: {route}, agents: {active_agents}")

    # Conversational / greeting queries skip retrieval entirely
    if route == "direct_chat" or "direct_chat" in active_agents:
        return "direct_chat_agent"

    # Multi-domain queries go through the supervisor
    if route == "multi" or len(active_agents) > 1:
        return "supervisor"

    domain = active_agents[0] if active_agents else "general"
    return {
        "finance": "finance_agent",
        "tech": "tech_agent",
        "web": "web_agent",
        "tools": "tool_agent",
    }.get(domain, "general_agent")


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
