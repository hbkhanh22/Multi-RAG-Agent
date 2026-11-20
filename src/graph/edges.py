"""
Graph Edges - Routing logic between nodes
"""

from typing import Literal
import logging

from .state import AgentState

logger = logging.getLogger(__name__)

NextNode = Literal[
    "supervisor", "sport_agent", "tech_agent", "general_agent",
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
        "sport": "sport_agent",
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


def route_after_evaluation(state: AgentState) -> Literal["refine", "__end__"]:
    """Revise the answer when the judge failed it and refine rounds remain."""
    verdict = state.get("evaluation") or {}
    if verdict.get("passed") is not False:  # passed, or not evaluated
        return "__end__"

    if state.get("iteration", 0) >= state.get("max_iterations", 0):
        logger.info("Answer still below threshold but no refine rounds left, ending")
        return "__end__"

    logger.info(f"Answer failed {verdict.get('failed_criteria')}, refining")
    return "refine"
