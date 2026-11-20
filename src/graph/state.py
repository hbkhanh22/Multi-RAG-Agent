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
    expanded_queries: List[str]  # multi-query rewrites computed while routing ([] = not computed)
    guardrail: Dict[str, Any]    # input guardrail verdict ({"blocked": True, ...} => refused)
    
    # ========== Routing ==========
    route: str  # "sport" | "tech" | "general" | "multi"
    active_agents: List[str]  # List of agent names to activate
    router_confidence: float
    
    # ========== Agent Responses ==========
    agent_responses: Annotated[Dict[str, Dict[str, Any]], merge_dicts]  # {agent_name: response_dict}
    retrieval_stats: Dict[str, Dict[str, int]]  # {agent_name: {retrieved, reranked}}
    
    # ========== Synthesis ==========
    final_answer: str
    key_themes: List[str]
    sources: List[Dict]
    contexts: List[Dict]  # every chunk the answering LLM saw, numbered like its [Source N] markers

    # ========== Self-evaluation (judge -> refine loop) ==========
    evaluation: Dict[str, Any]  # latest judge verdict ({} = not evaluated)
    evaluation_history: List[Dict[str, Any]]  # one verdict per round, oldest first

    # ========== Control & Metadata ==========
    iteration: int  # refine rounds done so far
    max_iterations: int  # refine rounds allowed (settings.self_eval_max_rounds)
    error: str
    status: str  # "pending" | "processing" | "complete" | "error"
    
    # ========== Performance Metrics ==========
    start_time: float
    processing_times: Dict[str, float]  # {component: time_in_seconds}
