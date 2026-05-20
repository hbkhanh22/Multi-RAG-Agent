"""
Graph State - State schema for LangGraph workflow
"""

from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State schema for multi-RAG agent graph"""
    
    # ========== Conversation ==========
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    
    # ========== Routing ==========
    route: str  # "finance" | "tech" | "general" | "multi"
    active_agents: List[str]  # List of agent names to activate
    router_confidence: float
    
    # ========== Retrieval Results ==========
    retrieved_docs: Dict[str, List[Dict]]  # {agent_name: [docs]}
    reranked_docs: Dict[str, List[Dict]]  # {agent_name: [docs]}
    
    # ========== Agent Responses ==========
    agent_responses: Dict[str, Dict[str, Any]]  # {agent_name: response_dict}
    
    # ========== Synthesis ==========
    final_answer: str
    key_themes: List[str]
    sources: List[Dict]
    
    # ========== Control & Metadata ==========
    iteration: int
    max_iterations: int
    error: str
    status: str  # "pending" | "processing" | "complete" | "error"
    
    # ========== Performance Metrics ==========
    start_time: float
    processing_times: Dict[str, float]  # {component: time_in_seconds}


class MessageState(TypedDict):
    """Simple state for message-based workflows"""
    messages: Annotated[List[BaseMessage], operator.add]
    

class RouterState(TypedDict):
    """State for router node"""
    query: str
    route: str
    agents: List[str]
    confidence: float


class RetrievalState(TypedDict):
    """State for retrieval nodes"""
    query: str
    domain: str
    retrieved_docs: List[Dict]
    reranked_docs: List[Dict]


class SynthesisState(TypedDict):
    """State for synthesis node"""
    query: str
    agent_responses: Dict[str, Dict]
    final_answer: str
    sources: List[Dict]