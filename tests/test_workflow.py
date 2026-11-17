"""End-to-end graph tests: fake LLM, real vector store / embeddings / reranker."""
from tests.conftest import MCP_QUERY, OFF_TOPIC_QUERY


def run(query):
    from src.graph import run_workflow
    return run_workflow(query)


def test_tech_question_uses_tech_docs(fake_llm, fake_web, tech_ready):
    r = run(MCP_QUERY)
    assert r["status"] == "complete"
    resp = r["agent_responses"]["tech"]
    assert resp["success"] and resp["fallback"] == "none"
    assert r["final_answer"] == "Docs say so [Source 1]"  # single agent: passthrough, no rewrite
    assert r["sources"] and r["sources"][0]["agent"] == "tech"
    assert "data/raw/tech" in r["sources"][0]["source"].replace("\\", "/")
    assert r["retrieval_stats"]["tech"]["reranked"] >= 1


def test_irrelevant_local_docs_fall_back_to_web(fake_llm, fake_web, tech_ready):
    """Regression: finance store is empty and tech docs are unrelated -> must use web, not tech docs."""
    fake_llm.route_json = {"route": "finance", "all_domains": ["finance"], "confidence": 0.9}
    r = run(OFF_TOPIC_QUERY)
    assert r["status"] == "complete"
    resp = r["agent_responses"]["finance"]
    assert resp["fallback"] == "web"
    assert all(s["url"].startswith("https://example.com") for s in r["sources"])


def test_multi_route_runs_both_agents_with_unique_source_ids(fake_llm, fake_web, tech_ready):
    fake_llm.route_json = {"route": "multi", "all_domains": ["tech", "web"], "confidence": 0.9}
    r = run(MCP_QUERY)
    assert r["status"] == "complete"
    assert set(r["agent_responses"]) == {"tech", "web"}
    assert r["final_answer"].startswith("MERGED") and r["key_themes"] == ["a", "b"]
    ids = [s["source_id"] for s in r["sources"]]
    assert len(ids) == len(set(ids)) and len(ids) >= 2


def test_direct_chat_skips_retrieval(fake_llm):
    fake_llm.route_json = {"route": "direct_chat", "all_domains": ["direct_chat"], "confidence": 1.0}
    r = run("Xin chào")
    assert r["status"] == "complete" and r["sources"] == [] and r["agent_responses"] == {}


def test_generation_failure_is_reported_as_error(fake_llm, fake_web, tech_ready):
    """Regression: status used to be 'complete' even when every agent failed."""
    fake_llm.fail_generation = True
    r = run(MCP_QUERY)
    assert r["status"] == "error"


def test_messages_not_duplicated(fake_llm, fake_web, tech_ready):
    """Regression: nodes returned the whole state, re-applying the `messages` reducer."""
    r = run(MCP_QUERY)
    assert len(r["messages"]) == 2  # user question + final AI answer


def test_greeting_skips_router_llm(fake_llm):
    """Fast path: small talk must not cost a router LLM call."""
    r = run("Xin chào")
    assert r["status"] == "complete" and r["final_answer"].startswith("Chào bạn")
    assert not any("query router" in c for c in fake_llm.calls)


def test_stream_emits_tokens_then_final(fake_llm, fake_web, tech_ready):
    from src.graph import stream_workflow
    events = list(stream_workflow(MCP_QUERY))
    types = [e["type"] for e in events]
    assert types[0] == "stage" and "route" in types and types[-1] == "final"
    streamed = "".join(e["text"] for e in events if e["type"] == "token")
    assert streamed == events[-1]["state"]["final_answer"] == "Docs say so [Source 1]"


def test_stream_direct_chat_tokens(fake_llm):
    from src.graph import stream_workflow
    events = list(stream_workflow("Xin chào"))
    assert "".join(e["text"] for e in events if e["type"] == "token") == events[-1]["state"]["final_answer"]
