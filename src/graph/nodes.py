"""
Graph Nodes - Node functions for LangGraph workflow.

Every node returns ONLY the keys it changes (a partial state update). Mutating
and returning the whole state would re-apply reducers (e.g. `messages` uses
operator.add) and duplicate data.
"""

import logging
from typing import Dict, Any

from langchain_core.messages import AIMessage

from src.agents import RouterAgent, RAGAgent, SupervisorAgent, SynthesizerAgent, WebAgent
from src.graph.state import AgentState
from src.utils import Timer

logger = logging.getLogger(__name__)

# Agents are created once and shared (the supervisor reuses the same instances).
router_agent = RouterAgent()
synthesizer_agent = SynthesizerAgent()
web_agent = WebAgent()
rag_agents = {
    "finance": RAGAgent("finance"),
    "tech": RAGAgent("tech"),
    "general": RAGAgent("general"),
}
supervisor_agent = SupervisorAgent(agents={**rag_agents, "web": web_agent})


def router_node(state: AgentState) -> Dict[str, Any]:
    """Route query to appropriate agents"""
    logger.info("=== ROUTER NODE ===")

    with Timer("Router"):
        result = router_agent.run({"query": state["query"]})

    route = result.get("route", "general")
    agents = result.get("agents", ["general"])
    logger.info(f"Route: {route}, Agents: {agents}")

    return {
        "route": route,
        "active_agents": agents,
        "router_confidence": result.get("confidence", 0.5),
    }


def _run_agent(name: str, agent, state: AgentState) -> Dict[str, Any]:
    with Timer(f"{name} agent"):
        result = agent.run({"query": state["query"]})
    logger.info(f"{name} result: success={result.get('success')}")
    return {"agent_responses": {name: result}}


def finance_rag_node(state: AgentState) -> Dict[str, Any]:
    logger.info("=== FINANCE RAG NODE ===")
    return _run_agent("finance", rag_agents["finance"], state)


def tech_rag_node(state: AgentState) -> Dict[str, Any]:
    logger.info("=== TECH RAG NODE ===")
    return _run_agent("tech", rag_agents["tech"], state)


def general_rag_node(state: AgentState) -> Dict[str, Any]:
    logger.info("=== GENERAL RAG NODE ===")
    return _run_agent("general", rag_agents["general"], state)


def web_agent_node(state: AgentState) -> Dict[str, Any]:
    """Web search agent node for live internet queries"""
    logger.info("=== WEB AGENT NODE ===")
    return _run_agent("web", web_agent, state)


def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """Run several specialist agents in parallel for multi-domain queries"""
    logger.info("=== SUPERVISOR NODE ===")

    with Timer("Supervisor"):
        result = supervisor_agent.run({
            "query": state["query"],
            "active_agents": state["active_agents"],
        })

    return {"agent_responses": result.get("agent_responses", {})}


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Build the final answer from agent responses"""
    logger.info("=== SYNTHESIZER NODE ===")

    agent_responses = state.get("agent_responses", {})
    if not agent_responses:
        logger.warning("No agent responses to synthesize")
        return {
            "final_answer": "Không có kết quả từ các agent.",
            "sources": [],
            "status": "error",
        }

    with Timer("Synthesizer"):
        result = synthesizer_agent.run({
            "query": state["query"],
            "agent_responses": agent_responses,
        })

    final_answer = result.get("synthesized_answer", "")
    logger.info(f"Synthesis complete. Answer length: {len(final_answer)}")

    return {
        "final_answer": final_answer,
        "key_themes": result.get("key_themes", []),
        "sources": result.get("sources", []),
        "retrieval_stats": result.get("retrieval_stats", {}),
        "status": "complete" if result.get("success") else "error",
        "messages": [AIMessage(content=final_answer)],
    }


def direct_chat_node(state: AgentState) -> Dict[str, Any]:
    """Handle conversational chat and greetings directly without document retrieval"""
    logger.info("=== DIRECT CHAT NODE ===")

    system_prompt = (
        "Bạn là trợ lý AI thông minh của hệ thống Multi-RAG Agent. "
        "Hãy chào hỏi hoặc trò chuyện một cách tự nhiên, lịch sự và thân thiện với người dùng (bằng cùng ngôn ngữ của người dùng). "
        "Cho người dùng biết bạn sẵn sàng giải đáp và tra cứu thông tin từ tài liệu trong các lĩnh vực: "
        "Tài chính (Finance), Công nghệ (Tech) và Kiến thức tổng quát (General), cũng như tìm kiếm thông tin mới trên Internet."
    )

    with Timer("Direct Chat"):
        try:
            response_text = router_agent.invoke([state["query"]], system_prompt=system_prompt)
            status = "complete"
        except Exception as e:
            logger.error(f"Direct chat failed: {e}")
            response_text = "Xin lỗi, hiện tôi không thể trả lời. Vui lòng thử lại sau."
            status = "error"

    return {
        "final_answer": response_text,
        "status": status,
        "sources": [],
        "key_themes": [],
        "messages": [AIMessage(content=response_text)],
    }
