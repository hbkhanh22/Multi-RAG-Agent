"""
Shared fixtures. The LLM is replaced by a deterministic fake; the vector stores,
embeddings and reranker are the real local ones (the `tech` store must be
ingested: data/raw/tech PDFs).
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MCP_QUERY = "What is Model Context Protocol (MCP)?"
OFF_TOPIC_QUERY = "What is the interest rate on 10-year US treasury bonds?"

FAKE_WEB_RESULTS = [
    {"title": "Treasury yields", "url": "https://example.com/a", "snippet": "10y yield is 4%",
     "source": "https://example.com/a", "source_id": 1, "domain": "web", "type": "web_search", "rank": 1},
    {"title": "Bond markets", "url": "https://example.com/b", "snippet": "Bonds rallied",
     "source": "https://example.com/b", "source_id": 2, "domain": "web", "type": "web_search", "rank": 2},
]


class FakeLLM:
    """Replaces BaseAgent.invoke. `route_json` is what the router 'decides'."""

    def __init__(self):
        self.route_json = {"route": "tech", "all_domains": ["tech"], "confidence": 0.9, "reasoning": "t"}
        self.fail_generation = False
        self.calls = []

    def __call__(self, agent, messages, system_prompt=None, on_token=None):
        out = self._answer(agent, messages, system_prompt)
        if on_token:  # emulate streaming: a few chunks
            for i in range(0, len(out), 8):
                on_token(out[i:i + 8])
        return out

    def _answer(self, agent, messages, system_prompt):
        sp = system_prompt or ""
        self.calls.append(sp[:40])
        if "query router" in sp:
            return json.dumps(self.route_json)
        if "merging answers" in sp:
            return json.dumps({"answer": "MERGED [Source 1] [Source 5]", "themes": ["a", "b"]})
        if self.fail_generation:
            raise RuntimeError("LLM down")
        if "Multi-RAG Agent" in sp and "chào hỏi" in sp:
            return "Chào bạn! Tôi có thể giúp gì?"
        if "Web Search Agent" in sp:
            return "Web says so [Source 1] [Source 2]"
        return "Docs say so [Source 1]"


@pytest.fixture(scope="session")
def modules():
    import src  # noqa: F401  (loads models once)
    from src.agents.base_agent import BaseAgent
    from src.retrieval import vector_store_manager, query_expander
    from src.tools.web_search import web_search_tool
    return {"BaseAgent": BaseAgent, "vsm": vector_store_manager,
            "expander": query_expander, "web": web_search_tool}


@pytest.fixture
def fake_llm(modules, monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(modules["BaseAgent"], "invoke",
                        lambda self, messages, system_prompt=None, on_token=None: llm(self, messages, system_prompt, on_token))
    monkeypatch.setattr(modules["expander"], "_invoke_llm", lambda messages: "")  # no expansion calls
    return llm


@pytest.fixture
def fake_web(modules, monkeypatch):
    monkeypatch.setattr(modules["web"], "search", lambda query, limit=None: list(FAKE_WEB_RESULTS))


@pytest.fixture(scope="session")
def tech_ready(modules):
    if modules["vsm"].get_store_stats("tech").get("document_count", 0) == 0:
        pytest.skip("tech vector store is empty (run the app once to ingest data/raw/tech)")
