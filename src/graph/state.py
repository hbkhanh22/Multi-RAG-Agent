"""
Graph State - State schema for LangGraph workflow
"""

from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langchain_core.messages import BaseMessage


def merge_dicts(left: Dict, right: Dict) -> Dict:
    """Reducer: merge two dicts so parallel/sequential nodes can add keys."""
    return {**(left or {}), **(right or {})}


class AgentState(TypedDict):
    """State schema for multi-RAG agent graph"""
    
    # ========== Conversation ==========
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    history: List[Dict[str, str]]  # earlier turns [{role, content}], oldest first
    standalone_query: str  # query rewritten to need no history (routing/retrieval use it)
    
    # ========== Routing ==========
    route: str  # "finance" | "tech" | "general" | "multi"
    active_agents: List[str]  # List of agent names to activate
    router_confidence: float
    
    # ========== Retrieval Results ==========
    retrieved_docs: Dict[str, List[Dict]]  # {agent_name: [docs]}
    reranked_docs: Dict[str, List[Dict]]  # {agent_name: [docs]}
    
    # ========== Agent Responses ==========
    agent_responses: Annotated[Dict[str, Dict[str, Any]], merge_dicts]  # {agent_name: response_dict}
    retrieval_stats: Dict[str, Dict[str, int]]  # {agent_name: {retrieved, reranked}}
    
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