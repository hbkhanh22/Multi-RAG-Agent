"""Pure-logic tests: router parsing, edges, calculator, synthesizer, scraper guard."""
import json

import pytest


@pytest.fixture(scope="module")
def router(modules):
    from src.agents.router_agent import RouterAgent
    r = RouterAgent.__new__(RouterAgent)  # parsing needs no LLM; skip __init__
    r.domains = ["sport", "tech", "general", "web"]
    return r


def parse(router, raw):
    return router._parse_classification_response(raw)


def test_router_parses_plain_json(router):
    r = parse(router, json.dumps({"route": "tech", "confidence": 0.9}))
    assert r["route"] == "tech" and r["agents"] == ["tech"] and r["confidence"] == 0.9


def test_router_parses_fenced_json(router):
    """Regression: missing `import re` made fenced JSON fall back to 'general'."""
    raw = '```json\n{"route": "sport", "confidence": 0.8}\n```'
    assert parse(router, raw)["route"] == "sport"


def test_router_parses_json_with_prose(router):
    raw = 'Sure! {"route": "web", "confidence": 0.7} Hope that helps'
    assert parse(router, raw)["route"] == "web"


def test_router_multi_filters_invalid_agents(router):
    r = parse(router, json.dumps({"route": "multi", "all_domains": ["tech", "bogus", "web", "tech"]}))
    assert r["route"] == "multi" and r["agents"] == ["tech", "web"]


def test_router_multi_with_single_domain_collapses(router):
    r = parse(router, json.dumps({"route": "multi", "all_domains": ["sport", "direct_chat"]}))
    assert r["route"] == "sport" and r["agents"] == ["sport"]


def test_router_unknown_route_defaults_general(router):
    assert parse(router, json.dumps({"route": "weather"}))["route"] == "general"


def test_router_garbage_defaults_general(router):
    assert parse(router, "no json here")["route"] == "general"


@pytest.mark.parametrize("route,agents,expected", [
    ("direct_chat", ["direct_chat"], "direct_chat_agent"),
    ("multi", ["tech", "web"], "supervisor"),
    ("sport", ["sport"], "sport_agent"),
    ("tech", ["tech"], "tech_agent"),
    ("web", ["web"], "web_agent"),
    ("general", ["general"], "general_agent"),
])
def test_edges(modules, route, agents, expected):
    from src.graph.edges import route_after_router
    assert route_after_router({"route": route, "active_agents": agents}) == expected


def test_calculator_basic_and_safety(modules):
    from src.tools.calculator_tool import calculator
    assert calculator.evaluate("2+2*3") == 8.0
    assert calculator.evaluate("sqrt(16)+max(1,2)") == 6.0
    assert calculator.evaluate("1/0") is None
    assert calculator.evaluate("__import__('os').system('x')") is None
    assert calculator.evaluate("9**9**9") is None  # must return fast, not hang


def test_scraper_blocks_private_hosts(modules):
    from src.tools.web_search import _is_public_url
    assert not _is_public_url("http://127.0.0.1:8000/")
    assert not _is_public_url("http://localhost/")
    assert not _is_public_url("http://192.168.0.1/")


@pytest.fixture
def synth(modules):
    from src.agents.synthesizer_agent import SynthesizerAgent
    return SynthesizerAgent.__new__(SynthesizerAgent)


def test_synthesizer_single_agent_passthrough(synth):
    resp = {"tech": {"success": True, "answer": "A [Source 2]", "retrieved_count": 9, "reranked_count": 4,
                     "sources": [{"source_id": 2, "source": "x.pdf", "page": 1}]}}
    out = synth.run({"query": "q", "agent_responses": resp})
    assert out["success"] and out["synthesized_answer"] == "A [Source 2]"
    assert out["sources"] == [{"source_id": 2, "source": "x.pdf", "page": 1, "agent": "tech"}]
    assert out["retrieval_stats"]["tech"] == {"retrieved": 9, "reranked": 4}
    assert not any(s.get("type") == "metadata" for s in out["sources"])  # stats are not "sources"


def test_synthesizer_renumbers_sources_globally(synth):
    usable = {
        "tech": {"answer": "T [Source 1]", "reranked_count": 4,
                 "sources": [{"source_id": 1, "source": "t.pdf"}]},
        "web": {"answer": "W [Source 1] [Source 2]", "reranked_count": 2,
                "sources": [{"source_id": 1, "source": "u1"}, {"source_id": 2, "source": "u2"}]},
    }
    answers, cites = synth._merge_with_global_numbering(usable)
    assert answers["tech"] == "T [Source 1]"
    assert answers["web"] == "W [Source 5] [Source 6]"
    assert [c["source_id"] for c in cites] == [1, 5, 6]


def test_synthesizer_all_failed(synth):
    out = synth.run({"query": "q", "agent_responses": {"tech": {"success": False, "answer": "boom"}}})
    assert out["success"] is False and "boom" in out["synthesized_answer"]


def test_synthesizer_json_parse_variants(synth):
    assert synth._parse_json('{"answer": "x"}')["answer"] == "x"
    assert synth._parse_json('```json\n{"answer": "y"}\n```')["answer"] == "y"
    assert synth._parse_json("nope") == {}


def test_query_expansion_parse_list(modules):
    from src.retrieval.query_expansion import _parse_string_list
    assert _parse_string_list('["a", "b"]') == ["a", "b"]
    assert _parse_string_list('```json\n["a", "b"]\n```') == ["a", "b"]
    assert _parse_string_list("not json") == []
    assert _parse_string_list('{"a": 1}') == []


def test_manifest_key_migration(modules, tmp_path, monkeypatch):
    from src.ingestion import auto_ingest as ai
    from config.settings import BASE_DIR
    manifest = tmp_path / "m.json"
    legacy_key = str(BASE_DIR / "data" / "raw" / "tech" / "x.pdf").replace("/", "\\")
    manifest.write_text(json.dumps({legacy_key: {"domain": "tech"}}), encoding="utf-8")
    monkeypatch.setattr(ai, "MANIFEST_FILE", manifest)
    assert list(ai._load_manifest()) == ["data/raw/tech/x.pdf"]


@pytest.mark.parametrize("q", ["Xin chào", "xin chào bạn!", "Hello", "hi", "Cảm ơn nhiều nhé", "Bạn là ai?", "Chào buổi sáng", "Tạm biệt"])
def test_fast_route_smalltalk(modules, q):
    from src.agents.router_agent import fast_route
    r = fast_route(q)
    assert r and r["route"] == "direct_chat" and r["agents"] == ["direct_chat"]


@pytest.mark.parametrize("q", ["MCP là gì?", "Xin chào, giải thích MCP giúp tôi", "Giá vàng hôm nay", "hi there, what is RAG?"])
def test_fast_route_ignores_real_questions(modules, q):
    from src.agents.router_agent import fast_route
    assert fast_route(q) is None
