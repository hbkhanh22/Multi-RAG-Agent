"""
Supervisor Agent - Runs several specialist agents in parallel for multi-domain
queries. Merging the answers is the Synthesizer's job (one LLM call), so the
supervisor itself makes no LLM calls.
"""

from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import contextvars
import logging

from .base_agent import BaseAgent
from .rag_agent import RAGAgent
from .web_agent import WebAgent
from config.settings import settings

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Coordinates multiple specialist RAG agents and the web agent"""

    def __init__(self, agents: Optional[Dict[str, Any]] = None):
        """
        Args:
            agents: Optional {name: agent} map to reuse existing instances.
                    If omitted, fresh specialists are created.
        """
        super().__init__(name="Supervisor Agent", model=settings.default_llm_model, temperature=0.0)

        self.agents = agents or {
            "sport": RAGAgent("sport"),
            "tech": RAGAgent("tech"),
            "general": RAGAgent("general"),
            "web": WebAgent(),
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the active agents in parallel.

        Args:
            input_data: Must contain 'query' and 'active_agents'. Optional 'history'
                (formatted earlier turns) and 'on_event' are passed to every agent.
        """
        try:
            query = input_data.get("query", "")
            active_agents = input_data.get("active_agents", ["general"])

            if not query:
                return {"success": False, "agent_responses": {}, "sources": []}

            shared = {k: input_data[k] for k in ("history", "on_event", "expanded_queries", "attachments")
                      if input_data.get(k)}
            agent_responses = self._execute_agents_parallel(query, active_agents, shared)

            return {
                "success": any(r.get("success") for r in agent_responses.values()),
                "agent_responses": agent_responses,
                "sources": self._extract_all_citations(agent_responses),
            }

        except Exception as e:
            logger.error(f"Error in supervisor: {e}", exc_info=True)
            return {"success": False, "agent_responses": {}, "sources": [], "error": str(e)}

    def _execute_agents_parallel(self, query: str, active_agents: List[str],
                                 shared: Optional[Dict[str, Any]] = None) -> Dict[str, Dict]:
        """Run each known agent in its own thread and collect the results.

        `shared` (history, on_event) is passed to every agent; no on_token: parallel
        answers are merged by the synthesizer, so they are not streamed."""
        names = []
        for name in dict.fromkeys(active_agents):  # dedupe, keep order
            if name in self.agents:
                names.append(name)
            else:
                logger.warning(f"Unknown agent skipped: {name}")

        def _run(name: str) -> Dict:
            try:
                return self.agents[name].run({"query": query, **(shared or {})})
            except Exception as e:
                logger.error(f"Error executing {name} agent: {e}", exc_info=True)
                return {"success": False, "error": str(e), "answer": f"Error in {name} agent", "sources": []}

        if not names:
            return {}

        # Each thread runs in a copy of the caller's context: LangGraph's stream writer
        # (and tracing callbacks) live in contextvars, which threads do not inherit.
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            futures = [pool.submit(contextvars.copy_context().run, _run, n) for n in names]
            results = [f.result() for f in futures]

        return dict(zip(names, results))

    def _extract_all_citations(self, responses: Dict[str, Dict]) -> List[Dict]:
        """Collect citations from all agent responses (copies, tagged with agent)"""
        citations = []
        for agent_name, response in responses.items():
            if response.get("success"):
                for citation in response.get("sources", []):
                    citations.append({**citation, "agent": agent_name})
        return citations
