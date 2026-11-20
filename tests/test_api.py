"""API tests (the app lifespan - and thus background ingestion - is not started)."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(modules):
    from api.main import app
    return TestClient(app)


def test_health_reports_components(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("healthy", "degraded")
    assert "vector_store_tech" in body["components"]


def test_invalid_domain_is_400_not_500(client):
    assert client.get("/api/domain/nope/stats").status_code == 400


def test_stats(client):
    r = client.get("/api/stats")
    assert r.status_code == 200 and set(r.json()["domains"]) == {"sport", "tech", "general"}


def test_query_validation(client):
    assert client.post("/api/query", json={"query": ""}).status_code == 422


def test_query_success_flag_follows_status(client, monkeypatch):
    import api.routes as routes
    monkeypatch.setattr(routes, "run_workflow", lambda q, debug=False, history=None, **kw: {
        "status": "error", "final_answer": "oops", "sources": [], "route": "tech"})
    body = client.post("/api/query", json={"query": "hi"}).json()
    assert body["success"] is False and body["final_answer"] == "oops"


def test_internal_errors_are_not_leaked(client, monkeypatch):
    import api.routes as routes

    def boom(q, debug=False, history=None):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(routes, "run_workflow", boom)
    r = client.post("/api/query", json={"query": "hi"})
    assert r.status_code == 500 and "secret" not in r.text and "trace id" in r.json()["detail"]


def test_batch_limit(client):
    r = client.post("/api/query/batch", json={"queries": ["q"] * 11})
    assert r.status_code == 422


def test_stream_endpoint_sse(client, monkeypatch):
    import json
    import api.routes as routes

    def fake_stream(q, history=None, **kw):
        yield {"type": "stage", "stage": "routing"}
        yield {"type": "token", "text": "Hi"}
        yield {"type": "final", "state": {"status": "complete", "final_answer": "Hi", "route": "direct_chat",
                                           "sources": [], "processing_times": {"total": 0.1}}}
    monkeypatch.setattr(routes, "stream_workflow", fake_stream)
    r = client.post("/api/query/stream", json={"query": "hi"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(b[6:]) for b in r.text.split("\n\n") if b.startswith("data: ")]
    assert [e["type"] for e in events] == ["start", "stage", "token", "final"]
    assert events[-1]["response"]["final_answer"] == "Hi" and events[-1]["response"]["success"] is True


def test_stream_endpoint_hides_errors(client, monkeypatch):
    import api.routes as routes
    monkeypatch.setattr(routes, "stream_workflow", lambda q, history=None, **kw: iter([{"type": "error", "message": "secret detail"}]))
    r = client.post("/api/query/stream", json={"query": "hi"})
    assert "secret" not in r.text and "trace id" in r.text
