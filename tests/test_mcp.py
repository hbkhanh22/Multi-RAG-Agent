"""MCP client: manager against a real stdio server, tool agent loop, routing, API."""
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

FAKE = str(Path(__file__).parent / "fake_mcp_server.py")


def fake_servers(**overrides):
    cfg = {"command": sys.executable, "args": [FAKE]}
    cfg.update(overrides)
    return {"fake": cfg}


@pytest.fixture(scope="module")
def manager(modules):
    from src.mcp_client import MCPManager
    m = MCPManager(servers=fake_servers(allowed_tools=["echo", "add", "boom", "big", "fetch"]))
    m.start(wait=True, timeout=60)
    yield m
    m.stop()


def test_manager_connects_and_lists_tools(manager):
    status = manager.status()
    assert status["ready"] and status["servers"]["fake"]["connected"]
    names = [s["function"]["name"] for s in manager.tool_specs()]
    assert "fake__echo" in names and "fake__add" in names
    spec = next(s for s in manager.tool_specs() if s["function"]["name"] == "fake__add")
    assert spec["function"]["parameters"]["type"] == "object"
    assert set(spec["function"]["parameters"]["properties"]) == {"a", "b"}


def test_call_tool(manager):
    assert manager.call_tool("fake__echo", {"text": "hi"}) == "echo:hi"
    assert manager.call_tool("fake__add", {"a": 2, "b": 3}) == "5"


def test_tool_error_and_unknown_tool(manager):
    from src.mcp_client import MCPToolError
    with pytest.raises(MCPToolError):
        manager.call_tool("fake__boom", {})
    with pytest.raises(MCPToolError, match="Unknown tool"):
        manager.call_tool("fake__nope", {})


def test_output_is_truncated(manager):
    from config.settings import settings
    out = manager.call_tool("fake__big", {})
    assert len(out) <= settings.mcp_max_output_chars + 20 and out.endswith("[truncated]")


def test_private_urls_blocked_before_reaching_server(manager):
    from src.mcp_client import MCPToolError
    with pytest.raises(MCPToolError, match="not a public"):
        manager.call_tool("fake__fetch", {"url": "http://127.0.0.1:8000/"})


def test_allowlist_hides_other_tools(modules):
    from src.mcp_client import MCPManager
    m = MCPManager(servers=fake_servers(allowed_tools=["echo"]))
    m.start(wait=True, timeout=60)
    try:
        assert [s["function"]["name"] for s in m.tool_specs()] == ["fake__echo"]
    finally:
        m.stop()


def test_failing_server_does_not_break_others(modules):
    from src.mcp_client import MCPManager
    servers = {**fake_servers(), "dead": {"command": sys.executable, "args": ["-c", "import sys; sys.exit(1)"]}}
    m = MCPManager(servers=servers)
    m.start(wait=True, timeout=60)
    try:
        st = m.status()["servers"]
        assert st["fake"]["connected"] and not st["dead"]["connected"] and st["dead"]["error"]
        assert m.has_tools()
    finally:
        m.stop()


# ------------------------------------------------------------------ tool agent

def scripted_agent(manager, replies):
    """ToolAgent whose LLM turns are scripted: each reply is an AIMessage."""
    from src.agents.tool_agent import ToolAgent
    agent = ToolAgent(manager=manager)
    it = iter(replies)
    seen = []

    def fake_chat(messages, specs, on_token):
        seen.append((list(messages), specs))
        msg = next(it)
        if msg.content and on_token:
            on_token(msg.content)
        return msg
    agent._chat = fake_chat
    agent.seen = seen
    return agent


def call(name, args, cid="c1"):
    return {"name": name, "args": args, "id": cid, "type": "tool_call"}


def test_tool_agent_loop(manager):
    agent = scripted_agent(manager, [
        AIMessage(content="", tool_calls=[call("fake__echo", {"text": "yo"})]),
        AIMessage(content="Kết quả là echo:yo"),
    ])
    tokens, events = [], []
    r = agent.run({"query": "echo yo", "on_token": tokens.append, "on_event": events.append})
    assert r["success"] and r["answer"] == "Kết quả là echo:yo" and "".join(tokens) == r["answer"]
    assert r["tool_calls"][0]["tool"] == "echo" and r["tool_calls"][0]["ok"]
    assert [e["status"] for e in events] == ["running", "done"]
    # the tool result was fed back to the model in the second turn
    assert any(getattr(m, "content", "") == "echo:yo" for m in agent.seen[1][0])


def test_tool_agent_reports_tool_failure_and_continues(manager):
    agent = scripted_agent(manager, [
        AIMessage(content="", tool_calls=[call("fake__boom", {})]),
        AIMessage(content="Công cụ bị lỗi."),
    ])
    r = agent.run({"query": "x"})
    assert r["success"] and r["tool_calls"][0]["ok"] is False
    assert "ERROR" in r["tool_calls"][0]["output_preview"]


def test_tool_agent_unknown_tool_name_is_an_error_not_a_crash(manager):
    agent = scripted_agent(manager, [
        AIMessage(content="", tool_calls=[call("fake__write_file", {"path": "x"})]),
        AIMessage(content="Không có công cụ đó."),
    ])
    r = agent.run({"query": "x"})
    assert r["success"] and r["tool_calls"][0]["ok"] is False


def test_tool_agent_round_limit_forces_answer(manager, monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "mcp_max_iterations", 2)
    loop_reply = lambda: AIMessage(content="", tool_calls=[call("fake__echo", {"text": "a"})])
    agent = scripted_agent(manager, [loop_reply(), loop_reply(), AIMessage(content="Tổng hợp cuối.")])
    r = agent.run({"query": "x"})
    assert r["success"] and r["answer"] == "Tổng hợp cuối." and len(r["tool_calls"]) == 2
    assert agent.seen[-1][1] == []  # final forced turn had no tools bound


def test_tool_agent_sources_from_urls_and_paths(manager):
    agent = scripted_agent(manager, [
        AIMessage(content="", tool_calls=[call("fake__fetch", {"url": "https://example.com/a"}, "1"),
                                          call("fake__echo", {"text": "no source"}, "2")]),
        AIMessage(content="ok"),
    ])
    r = agent.run({"query": "x"})
    assert [s["source"] for s in r["sources"]] == ["https://example.com/a"]
    assert r["sources"][0]["url"] == "https://example.com/a" and r["sources"][0]["agent"] == "tools"


def test_tool_agent_without_tools_fails_cleanly(modules):
    from src.agents.tool_agent import ToolAgent
    from src.mcp_client import MCPManager
    r = ToolAgent(manager=MCPManager(servers={})).run({"query": "đọc file"})
    assert r["success"] is False and "MCP" in r["answer"]


# --------------------------------------------------------------------- routing

def test_router_accepts_tools_route(modules):
    from src.agents.router_agent import RouterAgent
    r = RouterAgent.__new__(RouterAgent)
    r.domains = ["finance", "tech", "general", "web", "tools"]
    out = r._parse_classification_response('{"route": "tools", "confidence": 0.9}')
    assert out["route"] == "tools" and out["agents"] == ["tools"]


def test_edges_route_tools_to_tool_agent(modules):
    from src.graph.edges import route_after_router
    assert route_after_router({"route": "tools", "active_agents": ["tools"]}) == "tool_agent"


def test_tools_route_falls_back_to_web_when_no_servers(modules, monkeypatch):
    from src.agents.router_agent import RouterAgent
    r = RouterAgent.__new__(RouterAgent)
    r.domains = ["finance", "tech", "general", "web", "tools"]
    monkeypatch.setattr(RouterAgent, "_tools_available", staticmethod(lambda: False))
    monkeypatch.setattr(RouterAgent, "invoke", lambda self, msgs, system_prompt=None, **k: '{"route": "tools"}')
    out = r._classify_query("Đọc file hello.txt")
    assert out["route"] == "web" and out["agents"] == ["web"]


def test_router_prompt_mentions_tools_only_when_available(modules, monkeypatch):
    from src.agents.router_agent import RouterAgent
    r = RouterAgent.__new__(RouterAgent)
    seen = {}
    monkeypatch.setattr(RouterAgent, "invoke", lambda self, msgs, system_prompt=None, **k: seen.update(p=system_prompt) or '{"route": "web"}')
    monkeypatch.setattr(RouterAgent, "_tools_available", staticmethod(lambda: True))
    r.domains = ["finance", "tech", "general", "web", "tools"]
    r._classify_query("q")
    assert "- tools:" in seen["p"] and "{" in seen["p"] and "{tools_rule}" not in seen["p"] and "{domain" not in seen["p"]
    monkeypatch.setattr(RouterAgent, "_tools_available", staticmethod(lambda: False))
    r._classify_query("q")
    assert "- tools:" not in seen["p"] and "{tools_rule}" not in seen["p"]


# ------------------------------------------------------------------------- API

def test_mcp_status_endpoint(modules, monkeypatch):
    from fastapi.testclient import TestClient
    import api.routes as routes
    from api.main import app
    from src.mcp_client import MCPManager
    m = MCPManager(servers=fake_servers(allowed_tools=["echo"]))
    m.start(wait=True, timeout=60)
    monkeypatch.setattr(routes, "mcp_manager", m)
    try:
        body = TestClient(app).get("/api/mcp/status").json()
        assert body["tool_count"] == 1 and body["servers"]["fake"]["connected"]
        assert body["tools"][0]["name"] == "fake__echo"
        assert TestClient(app).get("/api/health").json()["components"]["mcp"] == "1 tools"
    finally:
        m.stop()


def test_tool_agent_prompt_is_read_only_and_lists_tools(manager):
    """Regression: the model once claimed to have created a file it had no tool for."""
    agent = scripted_agent(manager, [AIMessage(content="Tôi không thể tạo file.")])
    agent.run({"query": "tạo file moi.txt"})
    system = agent.seen[0][0][0].content
    assert "READ-ONLY" in system and "NEVER claim an action" in system and "fake__echo" in system
