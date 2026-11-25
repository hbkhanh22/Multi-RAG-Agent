"""Image Agent: policy + prompt, model fallback, storage, ownership, routing."""
import base64
import io
import json

import pytest

from config.settings import settings


def _jpeg(size=(64, 32)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, "green").save(buf, "JPEG")
    return buf.getvalue()


class _Resp:
    def __init__(self, artifact, status=200):
        self.artifact, self.status = artifact, status

    def raise_for_status(self):
        if self.status != 200:
            raise RuntimeError(self.status)

    def json(self):
        return {"artifacts": [self.artifact]}


OK = {"base64": base64.b64encode(_jpeg()).decode(), "finishReason": "SUCCESS"}


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{(tmp_path / 'img.db').as_posix()}")


def _http(monkeypatch, *responses):
    from src.images import generator
    calls, queue = [], list(responses)

    def post(url, **kw):
        calls.append((url, kw["json"]))
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(generator.httpx, "post", post)
    return calls


# ---------------------------------------------------------------- generator

def test_fallback_and_size(monkeypatch):
    from src.images import generate
    calls = _http(monkeypatch, TimeoutError(), _Resp(OK))
    img = generate("a cat", "landscape")
    assert img.model == settings.image_fallback_model and (img.width, img.height) == (64, 32)
    assert calls[0][0].endswith(settings.image_model) and calls[0][1]["width"] == 1344


def test_bad_answers(monkeypatch):
    from src.images import ContentFiltered, ImageUnavailable, generate
    _http(monkeypatch, _Resp({"base64": "AAAA", "finishReason": "SUCCESS"}), _Resp({}, 500))
    with pytest.raises(ImageUnavailable):
        generate("x")
    calls = _http(monkeypatch, _Resp({"base64": "", "finishReason": "CONTENT_FILTERED"}), _Resp(OK))
    with pytest.raises(ContentFiltered):
        generate("x")
    assert len(calls) == 1                            # a filtered prompt is not retried elsewhere


# -------------------------------------------------------------------- agent

PLAN = {"allowed": True, "prompt": "a cat next to a tree", "orientation": "square"}


@pytest.fixture
def agent(db, monkeypatch):
    from src.agents.base_agent import BaseAgent
    from src.agents.image_agent import ImageAgent

    monkeypatch.setattr(BaseAgent, "__init__", lambda self, name="", model=None, **kw: setattr(self, "model", model))
    a = ImageAgent()
    a.replies, a.prompts = [json.dumps(PLAN)], []

    def invoke(self, messages, system_prompt=None, on_token=None):
        a.prompts.append(messages[0])
        return a.replies[min(len(a.prompts), len(a.replies)) - 1]
    monkeypatch.setattr(BaseAgent, "invoke", invoke)
    return a


def test_draws_and_stores(agent, monkeypatch):
    from src.images import image_store
    calls, events = _http(monkeypatch, _Resp(OK)), []
    r = agent.run({"query": "vẽ mèo", "on_event": events.append})
    assert r["success"] and "vẽ mèo" in r["answer"] and events[-1]["stage"] == "drawing"
    assert image_store.load(r["images"][0]["id"], None)["data"] == _jpeg()
    assert calls[0][1]["prompt"] == PLAN["prompt"] and "data" not in json.dumps(r["images"])


@pytest.mark.parametrize("reply,refused", [
    ('{"allowed": false}', True), ("", False), ("sure!", False), ('{"prompt": "x"}', False),
    ('{"allowed": "true", "prompt": "x"}', False), ('{"allowed": true, "prompt": ""}', False),
])
def test_nothing_is_drawn_without_a_clear_permission(agent, monkeypatch, reply, refused):
    calls = _http(monkeypatch)
    agent.replies = [reply]
    r = agent.run({"query": "x"})
    assert calls == [] and r["images"] == [] and r["success"] is refused


def test_hourly_limit(agent, monkeypatch):
    monkeypatch.setattr(settings, "image_limit_per_hour", 1)
    calls = _http(monkeypatch, _Resp(OK))
    assert agent.run({"query": "x"})["images"]
    r = agent.run({"query": "x"})
    assert not r["success"] and "giới hạn" in r["answer"] and len(calls) == 1


def test_cancelled_image_is_not_stored(agent, monkeypatch):
    from src import runtime
    from src.images import generator, image_store
    ctx = runtime.RunContext("r1", owner="u")
    monkeypatch.setattr(runtime, "current", lambda: ctx)
    monkeypatch.setattr(generator.httpx, "post", lambda *a, **k: ctx.cancel() or _Resp(OK))
    with pytest.raises(runtime.Cancelled):
        agent.run({"query": "x"})
    assert image_store.recent_count(None) == 0


# ---------------------------------------------------------------------- API

def test_image_is_private_and_deleted_with_the_chat(db, monkeypatch):
    from fastapi.testclient import TestClient
    import api.routes as routes
    from api.main import app
    from src.images import image_store
    from src.images.generator import GeneratedImage
    from src.memory.store import ConversationStore

    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "jwt_secret", "s" * 40)
    monkeypatch.setattr(routes, "conversation_store", ConversationStore(settings.database_url))
    c = TestClient(app)

    def user(n):
        j = c.post("/api/auth/register", json={"email": f"{n}@example.com", "password": "correct horse"}).json()
        return j["user"]["id"], {"Authorization": "Bearer " + j["access_token"]}
    (a, ha), (_, hb) = user("a"), user("b")
    iid = image_store.add(GeneratedImage(_jpeg(), "image/jpeg", 64, 32, "m"), "p", a)
    monkeypatch.setattr(routes, "run_workflow", lambda q, **k: {
        "status": "complete", "final_answer": "ok", "route": "image",
        "agent_responses": {"image": {"images": [{"id": iid, "width": 64, "height": 32, "alt": "x"}]}}})
    r = c.post("/api/query", json={"query": "vẽ"}, headers=ha).json()
    assert r["images"][0]["id"] == iid
    cid = r["conversation_id"]
    assert c.get(f"/api/conversations/{cid}", headers=ha).json()["messages"][1]["meta"]["images"][0]["id"] == iid
    got = c.get(f"/api/images/{iid}", headers=ha)
    assert got.content == _jpeg() and got.headers["content-type"] == "image/jpeg"
    assert c.get(f"/api/images/{iid}", headers=hb).status_code == 404
    assert c.get(f"/api/images/{iid}").status_code == 401
    c.delete(f"/api/conversations/{cid}", headers=ha)
    assert c.get(f"/api/images/{iid}", headers=ha).status_code == 404


# -------------------------------------------------------------------- graph

@pytest.fixture
def drawn(monkeypatch):
    from src.graph import nodes
    seen = []

    def run(data):
        seen.append(data)
        return {"success": True, "answer": "caption", "images": [{"id": "i1"}], "sources": []}
    monkeypatch.setattr(nodes.image_agent, "run", run)
    return seen


def test_image_route(fake_llm, drawn):
    from src.graph import run_workflow
    fake_llm.route_json = {"route": "image", "all_domains": ["image"]}
    r = run_workflow("Bạn hãy vẽ cho tôi 1 con mèo bên cạnh cái cây")
    assert r["route"] == "image" and r["final_answer"] == "caption" and r["status"] == "complete" and not r["evaluation"]


def test_image_next_to_an_answer(fake_llm, fake_web, drawn):
    from src.graph import run_workflow
    fake_llm.route_json = {"route": "multi", "all_domains": ["web", "image"]}
    r = run_workflow("Tin mới nhất và vẽ minh họa")
    assert r["agent_responses"]["image"]["images"] and r["final_answer"].startswith("Web says so")
    assert "Web says so" in drawn[0]["material"]      # drawn after the answer, knowing what it says


def test_image_route_can_be_switched_off(fake_llm, monkeypatch):
    from src.graph.nodes import router_agent
    fake_llm.route_json = {"route": "image", "all_domains": ["image"]}
    monkeypatch.setattr(settings, "image_enabled", False)
    assert router_agent.run({"query": "vẽ cho tôi một con mèo"})["agents"] == ["direct_chat"]


def test_diagrams_are_not_pictures(fake_llm):
    from src.graph.nodes import router_agent
    fake_llm.route_json = {"route": "image", "all_domains": ["image"]}
    for q in ("Vẽ sơ đồ kiến trúc RAG bằng mermaid", "ve bieu do doanh thu", "Draw a flowchart of login"):
        assert router_agent.run({"query": q})["agents"] == ["tech"], q
    for q in ("Vẽ con mèo", "Vẽ ảnh một cô giáo chỉ vào biểu đồ"):
        assert router_agent.run({"query": q})["agents"] == ["image"], q


def test_missing_picture_is_mentioned(fake_llm, fake_web, monkeypatch):
    from src.graph import nodes, run_workflow
    monkeypatch.setattr(nodes.image_agent, "run", lambda d: {"success": True, "refused": True, "answer": "Không vẽ được.", "images": []})
    fake_llm.route_json = {"route": "multi", "all_domains": ["web", "image"]}
    assert run_workflow("Tin và vẽ minh họa")["final_answer"].endswith("*Không vẽ được.*")


def test_refused_picture_is_not_remembered(db, monkeypatch):
    import api.routes as routes
    from src.memory.store import ConversationStore
    store = ConversationStore(settings.database_url)
    monkeypatch.setattr(routes, "conversation_store", store)
    cid = store.create_conversation()
    resp = routes._build_response("x", {"status": "complete", "final_answer": "no", "route": "image",
                                        "agent_responses": {"image": {"refused": True, "images": []}}})
    routes._remember(cid, "x", resp)
    assert store.get_messages(cid) == []
