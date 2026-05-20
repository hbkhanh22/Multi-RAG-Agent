"""
Graph Nodes - Node functions for LangGraph workflow
"""

import logging
import time
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage
from src.agents import RouterAgent, RAGAgent, SupervisorAgent, SynthesizerAgent
from src.graph.state import AgentState
from src.utils import Timer

logger = logging.getLogger(__name__)

# Initialize agents
router_agent = RouterAgent()
supervisor_agent = SupervisorAgent()
synthesizer_agent = SynthesizerAgent()
rag_agents = {
    "finance": RAGAgent("finance"),
    "tech": RAGAgent("tech"),
    "general": RAGAgent("general")
}


def router_node(state: AgentState) -> Dict[str, Any]:
    """Route query to appropriate agents"""
    logger.info("=== ROUTER NODE ===")
    
    with Timer("Router"):
        query = state["query"]
        
        # Route query
        result = router_agent.run({"query": query})
        
        state["route"] = result.get("route", "general")
        state["active_agents"] = result.get("agents", ["general"])
        state["router_confidence"] = result.get("confidence", 0.5)
        
        logger.info(f"Route: {state['route']}, Agents: {state['active_agents']}")
        
        # Add to messages
        state["messages"].append(
            HumanMessage(content=f"Route: {state['route']}")
        )
    
    return state


def finance_rag_node(state: AgentState) -> Dict[str, Any]:
    """Finance RAG agent node"""
    logger.info("=== FINANCE RAG NODE ===")
    
    with Timer("Finance RAG"):
        query = state["query"]
        agent = rag_agents["finance"]
        
        result = agent.run({"query": query})
        
        if "agent_responses" not in state:
            state["agent_responses"] = {}
        
        state["agent_responses"]["finance"] = result
        logger.info(f"Finance result: {result.get('success')}")
    
    return state


def tech_rag_node(state: AgentState) -> Dict[str, Any]:
    """Tech RAG agent node"""
    logger.info("=== TECH RAG NODE ===")
    
    with Timer("Tech RAG"):
        query = state["query"]
        agent = rag_agents["tech"]
        
        result = agent.run({"query": query})
        
        if "agent_responses" not in state:
            state["agent_responses"] = {}
        
        state["agent_responses"]["tech"] = result
        logger.info(f"Tech result: {result.get('success')}")
    
    return state


def general_rag_node(state: AgentState) -> Dict[str, Any]:
    """General RAG agent node"""
    logger.info("=== GENERAL RAG NODE ===")
    
    with Timer("General RAG"):
        query = state["query"]
        agent = rag_agents["general"]
        
        result = agent.run({"query": query})
        
        if "agent_responses" not in state:
            state["agent_responses"] = {}
        
        state["agent_responses"]["general"] = result
        logger.info(f"General result: {result.get('success')}")
    
    return state


def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """Supervisor coordination node for complex queries"""
    logger.info("=== SUPERVISOR NODE ===")
    
    with Timer("Supervisor"):
        query = state["query"]
        active_agents = state["active_agents"]
        
        result = supervisor_agent.run({
            "query": query,
            "active_agents": active_agents
        })
        
        if "agent_responses" not in state:
            state["agent_responses"] = {}
        
        state["agent_responses"].update(result.get("agent_responses", {}))
        logger.info("Supervisor coordination complete")
    
    return state


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Synthesize final answer from agent responses"""
    logger.info("=== SYNTHESIZER NODE ===")
    
    with Timer("Synthesizer"):
        query = state["query"]
        agent_responses = state.get("agent_responses", {})
        
        if not agent_responses:
            logger.warning("No agent responses to synthesize")
            state["final_answer"] = "No results from agents"
            state["sources"] = []
            return state
        
        result = synthesizer_agent.run({
            "query": query,
            "agent_responses": agent_responses
        })
        
        state["final_answer"] = result.get("synthesized_answer", "")
        state["key_themes"] = result.get("key_themes", [])
        state["sources"] = result.get("sources", [])
        
        logger.info(f"Synthesis complete. Answer length: {len(state['final_answer'])}")
        
        # Add to messages
        state["messages"].append(
            AIMessage(content=state["final_answer"])
        )
    
    return state
