"""Query expansion runs once, concurrently with the router, and is reused by the agents."""
from tests.conftest import MCP_QUERY


def _count_expansions(monkeypatch, modules):
    expander = modules["expander"]
    calls = []
    original = expander.get_expanded_queries

    def counting(query, domain=None, expansion_types=None):
        calls.append(query)
        return [query, "alt one", "alt two"]

    monkeypatch.setattr(expander, "get_expanded_queries", counting)
    return calls, original


def test_expansion_done_once_while_routing(fake_llm, fake_web, tech_ready, modules, monkeypatch):
    calls, _ = _count_expansions(monkeypatch, modules)
    from src.graph import run_workflow
    r = run_workflow(MCP_QUERY)
    assert r["status"] == "complete"
    assert calls == [MCP_QUERY]                       # not repeated by the tech agent
    assert r["expanded_queries"] == [MCP_QUERY, "alt one", "alt two"]


def test_no_expansion_for_small_talk(fake_llm, modules, monkeypatch):
    calls, _ = _count_expansions(monkeypatch, modules)
    from src.graph import run_workflow
    r = run_workflow("Xin chào")
    assert r["status"] == "complete" and calls == []


def test_parallel_expansion_can_be_disabled(fake_llm, fake_web, tech_ready, modules, monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "parallel_expansion", False)
    calls, _ = _count_expansions(monkeypatch, modules)
    from src.graph import run_workflow
    r = run_workflow(MCP_QUERY)
    assert r["status"] == "complete" and r["expanded_queries"] == []
    assert calls == [MCP_QUERY]                       # the agent expanded by itself
