"""
Graph Nodes - Node functions for LangGraph workflow.

Every node returns ONLY the keys it changes (a partial state update). Mutating
and returning the whole state would re-apply reducers (e.g. `messages` uses
operator.add) and duplicate data.
"""

import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer

from src.agents.base_agent import BaseAgent
from src.agents import RouterAgent, RAGAgent, SupervisorAgent, SynthesizerAgent, WebAgent, ToolAgent, FileAgent
from src.graph.state import AgentState
from src.utils import Timer
from src.memory import condense_prompt, format_history, history_to_messages, needs_condense
from src.agents.router_agent import fast_route
from src.retrieval import query_expander
from src.evaluation.judge import AnswerJudge, AnswerRefiner
from src.guardrails import input_guard
from config.settings import settings

logger = logging.getLogger(__name__)



def emit(event: Dict[str, Any]) -> None:
    """Send a progress/token event to the stream consumer (no-op outside a streamed run)."""
    try:
        get_stream_writer()(event)
    except Exception:
        pass


class ChatAgent(BaseAgent):
    """Plain conversational agent for small talk (fast model, no retrieval)."""

    def __init__(self):
        super().__init__(
            name="Chat Agent",
            model=settings.chat_llm_model or settings.default_llm_model,
            temperature=0.5,
            max_tokens=settings.chat_max_tokens,
        )

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - unused
        return {}


# Agents are created once and shared (the supervisor reuses the same instances).
router_agent = RouterAgent()
synthesizer_agent = SynthesizerAgent()
web_agent = WebAgent()
rag_agents = {
    "sport": RAGAgent("sport"),
    "tech": RAGAgent("tech"),
    "general": RAGAgent("general"),
}
tool_agent = ToolAgent()
file_agent = FileAgent()
supervisor_agent = SupervisorAgent(agents={**rag_agents, "web": web_agent, "tools": tool_agent,
                                           "files": file_agent})
chat_agent = ChatAgent()
judge_agent = AnswerJudge()
refiner_agent = AnswerRefiner()


def _q(state: AgentState) -> str:
    """The query to route/retrieve with: history-resolved when available."""
    return state.get("standalone_query") or state["query"]


def _new_files(state: AgentState) -> List[Dict[str, Any]]:
    """Files sent with this very message."""
    return [a for a in state.get("attachments") or [] if a.get("new")]


def _agent_files(state: AgentState) -> List[Dict[str, Any]]:
    """What the File Agent reads: the files just sent, else those sent earlier."""
    return _new_files(state) or (state.get("attachments") or [])


def condense_node(state: AgentState) -> Dict[str, Any]:
    """Rewrite a follow-up question into a standalone one (skipped when not needed)."""
    query = state["query"]
    history = state.get("history") or []
    # with a new file, "this" / "it" means that file, not an earlier topic
    if _new_files(state) or not needs_condense(query, history) or fast_route(query):
        return {"standalone_query": query}

    emit({"type": "stage", "stage": "understanding"})
    try:
        with Timer("Condense"):
            rewritten = chat_agent.invoke([condense_prompt(query, history)]).strip().strip('"')
        # Guard against empty / runaway rewrites: fall back to the original
        if not rewritten or len(rewritten) > 4 * max(len(query), 40):
            rewritten = query
    except Exception as e:
        logger.warning(f"Condense failed, using original query: {e}")
        rewritten = query
    logger.info(f"Standalone query: {rewritten!r}")
    return {"standalone_query": rewritten}


def router_node(state: AgentState) -> Dict[str, Any]:
    """Route the query. In parallel: query expansion (independent of the route) and the
    input guardrail (safety classifier). An unsafe request is routed to `blocked`."""
    logger.info("=== ROUTER NODE ===")
    emit({"type": "stage", "stage": "routing"})

    query = _q(state)
    raw_query = state.get("query") or query          # guard what the user actually typed
    expanded: List[str] = []
    verdict = None
    files = state.get("attachments") or []
    new_files = _new_files(state)
    smalltalk = fast_route(query) is not None and not new_files
    run_guard = settings.guardrails_enabled and not smalltalk
    run_expansion = settings.parallel_expansion and not smalltalk

    with Timer("Router + guard + expansion"), ThreadPoolExecutor(max_workers=2) as pool:
        guard_fut = (pool.submit(contextvars.copy_context().run, input_guard.check, raw_query)
                     if run_guard else None)
        exp_fut = (pool.submit(contextvars.copy_context().run, query_expander.get_expanded_queries,
                               query, None, ["multi_query"]) if run_expansion else None)
        result = router_agent.run({"query": query, "attachments": [a["filename"] for a in files]})
        if exp_fut is not None:
            try:
                expanded = exp_fut.result()
            except Exception as e:  # the agents fall back to expanding themselves
                logger.warning(f"Parallel expansion failed: {e}")
        if guard_fut is not None:
            verdict = guard_fut.result()

    if verdict is not None and verdict.blocked:
        emit({"type": "route", "route": "blocked", "agents": []})
        return {"route": "blocked", "active_agents": [], "router_confidence": 1.0,
                "expanded_queries": [], "guardrail": {**verdict.public(), "blocked": True}}

    route = result.get("route", "general")
    agents = result.get("agents", ["general"])
    if new_files:
        # the user just sent a file: it is always read; other agents only join when asked for
        extra = [a for a in agents if a in ("sport", "tech", "general", "web")] if route == "multi" else []
        agents = ["files", *extra]
        route = "multi" if extra else "files"
    logger.info(f"Route: {route}, Agents: {agents}")
    emit({"type": "route", "route": route, "agents": agents})

    return {
        "route": route,
        "active_agents": agents,
        "router_confidence": result.get("confidence", 0.5),
        "expanded_queries": expanded,
        "guardrail": verdict.public() if verdict is not None else {},
    }


def blocked_node(state: AgentState) -> Dict[str, Any]:
    """Refuse an unsafe request (no retrieval, no generation)."""
    logger.info("=== BLOCKED (guardrail) ===")
    verdict = state.get("guardrail") or {}
    text = input_guard.REFUSAL.format(reason=verdict.get("reason") or "unsafe content")
    emit({"type": "token", "text": text})
    return {
        "final_answer": text,
        "status": "blocked",
        "sources": [],
        "key_themes": [],
        "messages": [AIMessage(content=text)],
    }


def _token_event(text: str, draft: bool = False) -> Dict[str, Any]:
    """Token event; `draft` marks text that may still be replaced before the final answer
    (the client must not present it as the answer)."""
    return {"type": "token", "text": text, **({"draft": True} if draft else {})}


def _run_agent(name: str, agent, state: AgentState) -> Dict[str, Any]:
    emit({"type": "stage", "stage": "reading_files" if name == "files" else "retrieving"})
    # The judge may have the answer rewritten; the tool loop streams text between tool calls
    draft = name == "tools" or (settings.self_eval_enabled and settings.self_eval_max_rounds > 0)
    with Timer(f"{name} agent"):
        # Single-agent route: the answer is passed through unchanged, so stream it live
        result = agent.run({
            "query": _q(state),
            "history": format_history(state.get("history") or []),
            "expanded_queries": state.get("expanded_queries") or [],
            "attachments": _agent_files(state) if name == "files" else [],
            "on_token": lambda t: emit(_token_event(t, draft)),
            "on_event": emit,
        })
    logger.info(f"{name} result: success={result.get('success')}")
    return {"agent_responses": {name: result}}


def sport_rag_node(state: AgentState) -> Dict[str, Any]:
    logger.info("=== SPORT RAG NODE ===")
    return _run_agent("sport", rag_agents["sport"], state)


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


def tool_agent_node(state: AgentState) -> Dict[str, Any]:
    """MCP tool-calling agent node (files, URL fetch...)"""
    logger.info("=== TOOL AGENT NODE ===")
    return _run_agent("tools", tool_agent, state)


def file_agent_node(state: AgentState) -> Dict[str, Any]:
    """Answer from the files attached to the conversation"""
    logger.info("=== FILE AGENT NODE ===")
    return _run_agent("files", file_agent, state)


def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """Run several specialist agents in parallel for multi-domain queries"""
    logger.info("=== SUPERVISOR NODE ===")
    emit({"type": "stage", "stage": "retrieving"})

    with Timer("Supervisor"):
        result = supervisor_agent.run({
            "query": _q(state),
            "active_agents": state["active_agents"],
            "attachments": _agent_files(state),
            "history": format_history(state.get("history") or []),
            "expanded_queries": state.get("expanded_queries") or [],
            "on_event": emit,
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

    if len(agent_responses) > 1:
        emit({"type": "stage", "stage": "synthesizing"})
    with Timer("Synthesizer"):
        result = synthesizer_agent.run({
            "query": _q(state),
            "agent_responses": agent_responses,
        })

    final_answer = result.get("synthesized_answer", "")
    logger.info(f"Synthesis complete. Answer length: {len(final_answer)}")

    return {
        "final_answer": final_answer,
        "key_themes": result.get("key_themes", []),
        "sources": result.get("sources", []),
        "contexts": result.get("contexts", []),
        "retrieval_stats": result.get("retrieval_stats", {}),
        "status": "complete" if result.get("success") else "error",
        "messages": [AIMessage(content=final_answer)],
    }


def evaluator_node(state: AgentState) -> Dict[str, Any]:
    """LLM-as-judge: score the answer against the context it was built from."""
    if not settings.self_eval_enabled or state.get("status") != "complete":
        return {}
    contexts = state.get("contexts") or []
    answer = state.get("final_answer") or ""
    if not answer or not contexts:
        # Nothing to check faithfulness against (tool answers, no documents found)
        return {}

    logger.info("=== EVALUATOR NODE ===")
    emit({"type": "stage", "stage": "evaluating"})
    with Timer("Evaluator"):
        verdict = judge_agent.evaluate(_q(state), answer, contexts)
    if verdict is None:
        return {}

    verdict = {**verdict, "round": state.get("iteration", 0)}
    logger.info(
        f"Judge round {verdict['round']}: scores={verdict['scores']} passed={verdict['passed']}"
    )
    emit({"type": "evaluation", "evaluation": verdict})
    return {
        "evaluation": verdict,
        "evaluation_history": [*(state.get("evaluation_history") or []), verdict],
    }


def _citations_for(answer: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Citation list for the [Source N] markers used in an answer."""
    cites = []
    for c in contexts:
        if f"[Source {c.get('source_id')}]" in answer:
            cite = {k: v for k, v in c.items() if k != "text" and v is not None}
            cite["snippet"] = (c.get("text") or "")[:200]
            cites.append(cite)
    return cites


def refine_node(state: AgentState) -> Dict[str, Any]:
    """Send the judge's feedback back to the LLM and replace the answer with its revision."""
    logger.info("=== REFINE NODE ===")
    rounds = state.get("iteration", 0) + 1
    emit({"type": "stage", "stage": "refining"})
    emit({"type": "reset"})  # the client discards the streamed draft
    contexts = state.get("contexts") or []

    try:
        with Timer("Refine"):
            revised = refiner_agent.refine(
                _q(state),
                state.get("final_answer", ""),
                contexts,
                state.get("evaluation") or {},
                on_token=lambda t: emit(_token_event(t, draft=True)),
            )
    except Exception as e:
        logger.warning(f"Refinement failed, keeping the previous answer: {e}")
        revised = ""

    if not revised:
        emit({"type": "reset"})
        emit(_token_event(state.get("final_answer", ""), draft=True))
        return {"iteration": rounds}
    return {
        "final_answer": revised,
        "sources": _citations_for(revised, contexts),
        "iteration": rounds,
    }


def direct_chat_node(state: AgentState) -> Dict[str, Any]:
    """Handle conversational chat and greetings directly without document retrieval"""
    logger.info("=== DIRECT CHAT NODE ===")

    system_prompt = (
        "Bạn là trợ lý AI thông minh của hệ thống Multi-RAG Agent. "
        "Hãy chào hỏi hoặc trò chuyện một cách tự nhiên, lịch sự và thân thiện với người dùng (bằng cùng ngôn ngữ của người dùng). "
        "Cho người dùng biết bạn sẵn sàng giải đáp và tra cứu thông tin từ tài liệu trong các lĩnh vực: "
        "Thể thao (Sport), Công nghệ (Tech) và Kiến thức tổng quát (General), cũng như tìm kiếm thông tin mới trên Internet."
    )

    with Timer("Direct Chat"):
        try:
            response_text = chat_agent.invoke(
                [*history_to_messages(state.get("history") or []), state["query"]],
                system_prompt=system_prompt,
                on_token=lambda t: emit({"type": "token", "text": t}),
            )
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
