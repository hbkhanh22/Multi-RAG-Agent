"""Conversation memory: store, history helpers, condense node, API flow."""
import pytest

from tests.conftest import MCP_QUERY


@pytest.fixture
def store(tmp_path):
    from src.memory.store import ConversationStore
    return ConversationStore(tmp_path / "m.db")


def test_store_roundtrip_and_title(store):
    cid = store.create_conversation()
    assert store.exists(cid) and not store.exists("nope")
    assert store.list_conversations() == []  # empty chats are hidden
    store.add_exchange(cid, "MCP là gì?", "Một giao thức [Source 1]", meta={"route": "tech", "sources": [{"source_id": 1}]})
    store.add_exchange(cid, "Còn cái thứ hai?", "Trả lời hai")
    msgs = store.get_messages(cid)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[1]["meta"]["sources"] == [{"source_id": 1}]
    assert store.list_conversations()[0]["title"] == "MCP là gì?"  # title from the first question only
    assert [m["content"] for m in store.get_recent(cid, 2)] == ["Còn cái thứ hai?", "Trả lời hai"]


def test_store_delete_cascades(store):
    cid = store.create_conversation()
    store.add_exchange(cid, "a", "b")
    assert store.delete_conversation(cid) and not store.delete_conversation(cid)
    assert store.get_messages(cid) == []


def test_history_strips_source_tags_and_respects_budget(modules, monkeypatch):
    from config.settings import settings
    from src.memory import format_history, history_to_messages
    hist = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "ans [Source 3] done"}]
    assert "[Source" not in format_history(hist) and "ans done" in format_history(hist)
    monkeypatch.setattr(settings, "memory_max_turns", 1)
    long = [{"role": "user", "content": f"u{i}"} if i % 2 == 0 else {"role": "assistant", "content": f"a{i}"} for i in range(10)]
    assert len(history_to_messages(long)) == 2  # only the last turn


def test_needs_condense(modules):
    from src.memory import needs_condense
    hist = [{"role": "user", "content": "MCP là gì?"}]
    assert not needs_condense("Hãy so sánh kiến trúc microservices và monolith về khả năng mở rộng và chi phí vận hành", hist)
    assert needs_condense("Còn cái thứ hai?", hist)
    assert needs_condense("giải thích kỹ hơn", hist)
    assert not needs_condense("Còn cái thứ hai?", [])  # no history => nothing to resolve


def test_condense_node_rewrites_followup(fake_llm, monkeypatch, modules):
    import src.graph.nodes as nodes
    monkeypatch.setattr(nodes.chat_agent, "invoke", lambda messages, system_prompt=None, on_token=None: "Cấu trúc thứ hai của MCP là gì?")
    state = {"query": "Còn cái thứ hai?", "history": [{"role": "user", "content": "MCP gồm mấy cấu trúc?"}]}
    assert nodes.condense_node(state)["standalone_query"] == "Cấu trúc thứ hai của MCP là gì?"
    assert nodes.condense_node({"query": "Còn cái thứ hai?", "history": []})["standalone_query"] == "Còn cái thứ hai?"
    assert nodes.condense_node({"query": "cảm ơn", "history": state["history"]})["standalone_query"] == "cảm ơn"  # small talk skipped


def test_condense_failure_falls_back_to_original(modules, monkeypatch):
    import src.graph.nodes as nodes

    def boom(*a, **k):
        raise RuntimeError("llm down")
    monkeypatch.setattr(nodes.chat_agent, "invoke", boom)
    state = {"query": "Còn cái đó?", "history": [{"role": "user", "content": "x"}]}
    assert nodes.condense_node(state)["standalone_query"] == "Còn cái đó?"


def test_workflow_uses_standalone_query_for_retrieval(fake_llm, fake_web, tech_ready, monkeypatch):
    import src.graph.nodes as nodes
    from src.graph import run_workflow
    monkeypatch.setattr(nodes.chat_agent, "invoke", lambda messages, system_prompt=None, on_token=None: MCP_QUERY)
    seen = {}
    real = nodes.router_agent.run
    monkeypatch.setattr(nodes.router_agent, "run", lambda d: seen.update(d) or real(d))
    hist = [{"role": "user", "content": "Nói về MCP"}, {"role": "assistant", "content": "MCP là giao thức"}]
    r = run_workflow("giải thích kỹ hơn", history=hist)
    assert seen["query"] == MCP_QUERY and r["status"] == "complete"
    assert r["messages"][0].content == "giải thích kỹ hơn"  # user still sees their own words


@pytest.fixture
def api_client(modules, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import api.routes as routes
    from src.memory.store import ConversationStore
    monkeypatch.setattr(routes, "conversation_store", ConversationStore(tmp_path / "api.db"))
    return TestClient(__import__("api.main", fromlist=["app"]).app), routes


def test_api_conversation_flow(api_client, monkeypatch):
    client, routes = api_client
    calls = []

    def fake_run(q, debug=False, history=None):
        calls.append(list(history or []))
        return {"status": "complete", "final_answer": f"A:{q}", "route": "tech", "sources": [], "processing_times": {"total": 1}}
    monkeypatch.setattr(routes, "run_workflow", fake_run)

    r1 = client.post("/api/query", json={"query": "q1"}).json()
    cid = r1["conversation_id"]
    assert cid and calls[0] == []
    client.post("/api/query", json={"query": "q2", "conversation_id": cid})
    assert [m["content"] for m in calls[1]] == ["q1", "A:q1"]  # history reached the workflow

    assert client.get("/api/conversations").json()["conversations"][0]["id"] == cid
    msgs = client.get(f"/api/conversations/{cid}").json()["messages"]
    assert len(msgs) == 4 and msgs[1]["meta"]["route"] == "tech"
    assert client.delete(f"/api/conversations/{cid}").status_code == 200
    assert client.get(f"/api/conversations/{cid}").status_code == 404


def test_api_unknown_conversation_id_starts_new(api_client, monkeypatch):
    client, routes = api_client
    monkeypatch.setattr(routes, "run_workflow", lambda q, debug=False, history=None: {
        "status": "complete", "final_answer": "x", "route": "tech", "sources": []})
    cid = client.post("/api/query", json={"query": "q", "conversation_id": "does-not-exist"}).json()["conversation_id"]
    assert cid and cid != "does-not-exist"


def test_failed_answers_not_remembered(api_client, monkeypatch):
    client, routes = api_client
    monkeypatch.setattr(routes, "run_workflow", lambda q, debug=False, history=None: {
        "status": "error", "final_answer": "oops", "route": "tech", "sources": []})
    cid = client.post("/api/query", json={"query": "q"}).json()["conversation_id"]
    assert client.get("/api/conversations").json()["conversations"] == []
    assert client.get(f"/api/conversations/{cid}").json()["messages"] == []
