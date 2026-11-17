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
    assert r.status_code == 200 and set(r.json()["domains"]) == {"finance", "tech", "general"}


def test_query_validation(client):
    assert client.post("/api/query", json={"query": ""}).status_code == 422


def test_query_success_flag_follows_status(client, monkeypatch):
    import api.routes as routes
    monkeypatch.setattr(routes, "run_workflow", lambda q, debug=False: {
        "status": "error", "final_answer": "oops", "sources": [], "route": "tech"})
    body = client.post("/api/query", json={"query": "hi"}).json()
    assert body["success"] is False and body["final_answer"] == "oops"


def test_internal_errors_are_not_leaked(client, monkeypatch):
    import api.routes as routes

    def boom(q, debug=False):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(routes, "run_workflow", boom)
    r = client.post("/api/query", json={"query": "hi"})
    assert r.status_code == 500 and "secret" not in r.text and "trace id" in r.json()["detail"]


def test_batch_limit(client):
    r = client.post("/api/query/batch", json={"queries": ["q"] * 11})
    assert r.status_code == 422
