"""
Phase 4 infrastructure: job queue, document sync, rate limiting, cancellation,
token accounting, ops endpoints, document API. Optional live tests:
  TEST_PG_URL=postgresql+psycopg://...  -> pgvector backend + Postgres job queue
  ONNX models present                   -> int8 runtime sanity checks
"""
import io
import os
import time

import pytest
from langchain_core.documents import Document


# ------------------------------------------------------------------ helpers

class FakeEmbeddings:
    """Deterministic 1024-d bag-of-characters vectors (no model needed)."""

    def _vec(self, text):
        v = [0.0] * 1024
        for i, ch in enumerate(text.lower()):
            v[(ord(ch) * 31 + i % 7) % 1024] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class MemoryBackend:
    """In-memory VectorBackend used to test the sync logic without models."""

    name = "memory"

    def __init__(self):
        self.chunks = {}   # (domain, filename) -> [Document]

    def add(self, domain, docs):
        for d in docs:
            self.chunks.setdefault((domain, d.metadata["filename"]), []).append(d)
        return len(docs)

    def replace_file(self, domain, filename, docs):
        self.chunks[(domain, filename)] = list(docs)
        return len(docs)

    def delete_file(self, domain, filename):
        return len(self.chunks.pop((domain, filename), []))

    def search(self, domain, query, k):
        return [d for (dom, _), ds in self.chunks.items() if dom == domain for d in ds][:k]

    def all_documents(self, domain):
        return self.search(domain, "", 10_000)

    def count(self, domain):
        return len(self.all_documents(domain))

    def fingerprint(self, domain):
        return self.count(domain)

    def healthy(self):
        return True


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Point the whole storage layer at an empty SQLite database."""
    from config.settings import settings
    url = f"sqlite:///{(tmp_path / 'infra.db').as_posix()}"
    monkeypatch.setattr(settings, "database_url", url)
    return url


@pytest.fixture
def sandbox(tmp_path, monkeypatch, fresh_db):
    """Local document storage in tmp + in-memory vector backend."""
    from config.settings import settings
    from src.retrieval import vector_store as vs
    from src.storage import objects

    raw = tmp_path / "raw"
    paths = {d: raw / d for d in ("sport", "tech", "general")}
    for p in paths.values():
        p.mkdir(parents=True)
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "sport_data_path", paths["sport"])
    monkeypatch.setattr(settings, "tech_data_path", paths["tech"])
    monkeypatch.setattr(settings, "general_data_path", paths["general"])
    monkeypatch.setattr(settings, "contextual_chunks", False)
    monkeypatch.setattr(objects, "_storage", objects.LocalStorage())

    backend = MemoryBackend()
    manager = vs.VectorStoreManager(backend=backend, embeddings=FakeEmbeddings())
    monkeypatch.setattr(vs, "_manager", manager)   # lazy singleton => no model is loaded
    return {"raw": paths, "backend": backend, "manager": manager}


# ------------------------------------------------------------------ job queue

def test_job_queue_lifecycle(fresh_db, monkeypatch):
    from config.settings import settings
    from src.storage import jobs

    monkeypatch.setattr(settings, "ingest_max_attempts", 2)
    a = jobs.enqueue("sync", {"reason": "x"})
    assert jobs.enqueue("sync", {"reason": "x"}) == a          # deduped while queued
    b = jobs.enqueue("delete", {"key": "k"})

    job = jobs.claim()
    assert job["id"] == a and job["attempts"] == 1
    assert jobs.get(a)["status"] == "running"
    assert jobs.fail(job, "boom") == "queued"                  # retried
    job = jobs.claim()
    assert job["id"] == a and job["attempts"] == 2
    assert jobs.fail(job, "boom") == "failed"                  # attempts exhausted

    job = jobs.claim()
    assert job["id"] == b
    jobs.complete(b, {"ok": 1})
    assert jobs.get(b)["status"] == "done" and jobs.get(b)["result"] == {"ok": 1}
    assert jobs.claim() is None
    assert [j["id"] for j in jobs.recent()][:2] == [b, a]


# ------------------------------------------------------------------ document sync

def test_sync_indexes_updates_and_removes(sandbox):
    from src.ingestion.auto_ingest import list_documents, sync_documents

    f = sandbox["raw"]["general"] / "info.txt"
    f.write_text("Lưu Mai Hậu là sinh viên năm cuối.", encoding="utf-8")
    stats = sync_documents()
    assert stats["general"] == 1 and stats["failed"] == 0
    docs = list_documents()
    assert [(d["filename"], d["status"]) for d in docs] == [("info.txt", "ready")]
    assert "sinh viên" in sandbox["backend"].chunks[("general", "info.txt")][0].page_content

    assert sync_documents()["unchanged"] == 1                  # nothing changed => no re-embed

    time.sleep(0.01)
    f.write_text("Lưu Mai Hậu là kỹ sư phần mềm.", encoding="utf-8")
    os.utime(f, (time.time() + 5, time.time() + 5))
    assert sync_documents()["general"] == 1
    chunks = sandbox["backend"].chunks[("general", "info.txt")]
    assert len(chunks) == 1 and "kỹ sư" in chunks[0].page_content   # replaced, not duplicated

    f.unlink()
    assert sync_documents()["removed"] == 1
    assert list_documents() == [] and ("general", "info.txt") not in sandbox["backend"].chunks


def test_sync_marks_broken_files_failed_once(sandbox):
    from src.ingestion.auto_ingest import list_documents, sync_documents

    (sandbox["raw"]["tech"] / "broken.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    assert sync_documents()["failed"] == 1
    doc = list_documents()[0]
    assert doc["status"] == "failed" and doc["error"]
    assert sync_documents()["failed"] == 0                     # same broken file is not retried


def test_bm25_sees_changes_from_other_processes(sandbox):
    m, backend = sandbox["manager"], sandbox["backend"]
    filler = [Document(page_content=f"filler text {i}", metadata={"filename": f"f{i}.txt"}) for i in range(4)]
    backend.add("tech", filler + [Document(page_content="alpha beta", metadata={"filename": "a.txt"})])
    assert m.bm25_search("tech", "alpha")
    # another process (worker) writes directly to the backend: no local invalidation
    backend.add("tech", [Document(page_content="gamma delta", metadata={"filename": "b.txt"})])
    m._bm25_checked["tech"] = 0                                # TTL elapsed
    assert m.bm25_search("tech", "gamma")


def test_legacy_manifest_imported_once(sandbox, monkeypatch):
    import json
    from config.settings import settings
    from src.ingestion import auto_ingest as ai

    monkeypatch.setattr(settings, "vector_backend", "chroma")
    f = sandbox["raw"]["sport"] / "laws.txt"
    f.write_text("offside", encoding="utf-8")
    st = f.stat()
    manifest = ai._legacy_manifest_file()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    key = ai.get_storage().list_files()[0].key
    manifest.write_text(json.dumps({key: {"domain": "sport", "filename": "laws.txt", "mtime": st.st_mtime,
                                          "size": st.st_size, "chunks_count": 3}}), encoding="utf-8")
    stats = ai.sync_documents()
    assert stats["unchanged"] == 1 and stats["sport"] == 0      # already embedded: not redone
    assert ai.list_documents()[0]["chunks_count"] == 3


# ------------------------------------------------------------------ guards

def test_sliding_window_limiter():
    from api.deps import SlidingWindowLimiter

    lim = SlidingWindowLimiter(limit=2, window=60)
    assert lim.check("a") == 0 and lim.check("a") == 0
    assert lim.check("a") > 0                                   # third call in the window
    assert lim.check("b") == 0                                  # other clients unaffected
    assert SlidingWindowLimiter(limit=0).check("a") == 0        # disabled


def test_rate_limit_returns_429(modules, fake_llm, monkeypatch):
    from fastapi.testclient import TestClient
    from api import deps
    from api.main import app

    monkeypatch.setattr(deps, "query_limiter", deps.SlidingWindowLimiter(limit=1))
    c = TestClient(app)
    assert c.post("/api/route", json={"query": "hi"}).status_code == 200   # not limited
    r1 = c.post("/api/query", json={"query": "Xin chào"})
    r2 = c.post("/api/query", json={"query": "Xin chào"})
    assert r1.status_code == 200 and r2.status_code == 429 and "Retry-After" in r2.headers


# ------------------------------------------------------------------ cancellation + usage

def test_streaming_llm_call_stops_when_cancelled(modules, monkeypatch):
    from src import runtime
    from src.agents.base_agent import BaseAgent

    class Chunk:
        def __init__(self, text):
            self.content, self.usage_metadata = text, None

    class Agent(BaseAgent):
        def run(self, input_data):
            return {}

    agent = Agent("t")
    ctx = runtime.RunContext("req1")
    received = []

    def on_token(t):
        received.append(t)
        if len(received) == 2:
            ctx.cancel()

    monkeypatch.setattr(type(agent.llm), "stream", lambda self, msgs, **kw: (Chunk(f"t{i} ") for i in range(100)))
    token = runtime.bind(ctx)
    try:
        with pytest.raises(runtime.Cancelled):
            agent.invoke(["question"], on_token=on_token)
    finally:
        runtime._fallback.reset(token)
    assert len(received) == 2                                  # stopped right after the cancel


def test_usage_is_recorded_per_request(modules, monkeypatch):
    from src import runtime
    from src.agents.base_agent import BaseAgent

    class Msg:
        content = "answer"
        usage_metadata = {"input_tokens": 120, "output_tokens": 30}

    class Agent(BaseAgent):
        def run(self, input_data):
            return {}

    agent = Agent("t")
    monkeypatch.setattr(type(agent.llm), "invoke", lambda self, msgs, **kw: Msg())
    ctx = runtime.RunContext()
    token = runtime.bind(ctx)
    try:
        agent.invoke(["q"])
        agent.invoke(["q"])
    finally:
        runtime._fallback.reset(token)
    u = ctx.usage()
    assert (u["prompt_tokens"], u["completion_tokens"], u["llm_calls"], u["estimated"]) == (240, 60, 2, False)


def test_stream_emits_start_and_final_with_usage(modules, fake_llm):
    import json
    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app).stream("POST", "/api/query/stream", json={"query": "Xin chào"}) as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert events[0]["type"] == "start" and events[0]["request_id"]
    final = events[-1]
    assert final["type"] == "final" and final["response"]["request_id"] == events[0]["request_id"]
    assert "usage" in final["response"]
    assert r.headers["x-request-id"]


def test_workflow_cancelled_before_start(modules, fake_llm):
    from src import runtime
    from src.graph import stream_workflow

    ctx = runtime.RunContext()
    ctx.cancel()
    events = list(stream_workflow("What is MCP?", [], run_ctx=ctx))
    assert events[-1]["type"] == "cancelled"
    assert not any(e["type"] == "final" for e in events)


def test_cancel_unknown_request_is_404(modules):
    from fastapi.testclient import TestClient
    from api.main import app

    assert TestClient(app).post("/api/query/nope/cancel").status_code == 404


# ------------------------------------------------------------------ ops endpoints

def test_ops_endpoints(modules):
    from fastapi.testclient import TestClient
    from api.main import app

    c = TestClient(app)
    assert c.get("/health/live").json() == {"status": "alive"}
    r = c.get("/health/ready")
    assert r.status_code in (200, 503) and set(r.json()["checks"]) >= {"database", "vector_store", "storage"}
    assert r.json()["checks"]["database"] is True
    c.get("/api/health")
    m = c.get("/metrics")
    assert m.status_code == 200 and "rag_http_requests_total" in m.text


# ------------------------------------------------------------------ document API

def test_document_upload_list_delete(modules, sandbox, monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    from src.storage import jobs
    from src.worker import run_once

    c = TestClient(app)
    r = c.post("/api/documents", data={"domain": "general"},
               files={"file": ("../../evil name.txt", io.BytesIO("Hậu thích bóng đá.".encode()), "text/plain")})
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["key"].endswith("raw/general/evil name.txt")    # path traversal stripped
    assert c.get("/api/documents").json()["by_status"] == {"pending": 1}

    while run_once():                                           # the worker processes the queue
        pass
    docs = c.get("/api/documents").json()["documents"]
    assert docs[0]["status"] == "ready" and docs[0]["chunks_count"] == 1
    assert c.get(f"/api/jobs/{body['job_id']}").json()["status"] == "done"

    assert c.delete(f"/api/documents/{docs[0]['id']}").status_code == 202
    while run_once():
        pass
    assert c.get("/api/documents").json()["total"] == 0
    assert not (sandbox["raw"]["general"] / "evil name.txt").exists()
    assert jobs.claim() is None


def test_document_upload_validation_and_admin_token(modules, sandbox, monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    from config.settings import settings

    c = TestClient(app)
    bad = c.post("/api/documents", data={"domain": "general"},
                 files={"file": ("x.exe", io.BytesIO(b"MZ"), "application/octet-stream")})
    assert bad.status_code == 415
    assert c.post("/api/documents", data={"domain": "finance"},
                  files={"file": ("a.txt", io.BytesIO(b"x"), "text/plain")}).status_code == 422

    monkeypatch.setattr(settings, "admin_token", "s3cret")
    assert c.get("/api/documents").status_code == 401
    assert c.get("/api/documents", headers={"Authorization": "Bearer s3cret"}).status_code == 200


# ------------------------------------------------------------------ optional live backends

@pytest.mark.skipif(not os.environ.get("TEST_PG_URL"), reason="set TEST_PG_URL to test pgvector")
def test_pgvector_backend_roundtrip():
    from sqlalchemy import delete
    from src.retrieval.vector_store import PgVectorBackend, VectorStoreManager
    from src.storage.db import chunks, get_engine

    url = os.environ["TEST_PG_URL"]
    backend = PgVectorBackend(FakeEmbeddings(), "pytest-index", url=url)
    with get_engine(url).begin() as conn:
        conn.execute(delete(chunks).where(chunks.c.index_name == "pytest-index"))
    m = VectorStoreManager(backend=backend)

    docs = [Document(page_content=t, metadata={"filename": "f.txt", "source": "s", "page": i})
            for i, t in enumerate(["offside rule explained", "throw-in procedure", "goal kick"])]
    assert m.replace_file("sport", "f.txt", docs) == 3
    assert m.get_store_stats("sport")["document_count"] == 3
    assert m.search("sport", "offside rule", k=1)[0].page_content == "offside rule explained"
    assert m.search("tech", "offside rule", k=3) == []          # domain filter
    assert m.bm25_search("sport", "throw")[0].page_content == "throw-in procedure"
    assert m.replace_file("sport", "f.txt", docs[:1]) == 1      # atomic swap
    assert backend.count("sport") == 1
    assert m.delete_by_filename("sport", "f.txt") == 1


@pytest.mark.skipif(not os.environ.get("TEST_PG_URL"), reason="set TEST_PG_URL to test the Postgres queue")
def test_postgres_queue_skip_locked(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from config.settings import settings
    from src.storage import jobs

    monkeypatch.setattr(settings, "database_url", os.environ["TEST_PG_URL"])
    ids = {jobs.enqueue("sync", {"n": i}, dedupe=False) for i in range(6)}
    with ThreadPoolExecutor(6) as pool:
        claimed = [j for j in pool.map(lambda _: jobs.claim(), range(6)) if j]
    got = [j["id"] for j in claimed]
    assert len(got) == len(set(got))                            # no job taken twice
    for j in claimed:
        jobs.complete(j["id"])
    assert ids >= set(got)


def _onnx_ready():
    try:
        from config.settings import settings
        from src.retrieval.onnx_models import onnx_available
        return onnx_available(settings.reranker_model) and onnx_available(settings.embedding_model)
    except Exception:
        return False


@pytest.mark.skipif(not _onnx_ready(), reason="run scripts/export_onnx.py first")
def test_onnx_int8_models_rank_sensibly():
    import numpy as np
    from src.retrieval.onnx_models import OnnxCrossEncoder, OnnxEmbeddings

    q = "Vòng tròn giữa sân có bán kính bao nhiêu?"
    passages = ["A circle with a radius of 9.15 m is marked around the centre mark.",
                "MCP uses JSON-RPC 2.0 messages."]
    scores = OnnxCrossEncoder().predict([(q, p) for p in passages])
    assert scores[0] > scores[1] and 0 <= scores.min() and scores.max() <= 1
    emb = OnnxEmbeddings()
    qv, pv = np.array(emb.embed_query(q)), np.array(emb.embed_documents(passages))
    assert abs(np.linalg.norm(qv) - 1) < 1e-3 and (pv @ qv)[0] > (pv @ qv)[1]


# ------------------------------------------------------------------ S3 storage (moto)

def test_s3_storage_sync_roundtrip(sandbox, monkeypatch):
    """Upload -> sync -> modify (new ETag) -> delete, against an in-process fake S3."""
    moto = pytest.importorskip("moto")
    from config.settings import settings
    from src.ingestion.auto_ingest import list_documents, sync_documents
    from src.storage import objects

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setattr(settings, "s3_endpoint_url", "")
    monkeypatch.setattr(settings, "s3_bucket", "rag-test")
    with moto.mock_aws():
        s3 = objects.S3Storage()
        s3.ensure_bucket()
        monkeypatch.setattr(objects, "_storage", s3)

        key = s3.put("sport", "laws.txt", io.BytesIO("Việt vị là gì".encode()))
        assert key == "sport/laws.txt"
        files = s3.list_files()
        assert [(f.domain, f.filename) for f in files] == [("sport", "laws.txt")] and files[0].etag

        assert sync_documents()["sport"] == 1
        assert list_documents()[0]["etag"] == files[0].etag
        assert sync_documents()["unchanged"] == 1

        s3.put("sport", "laws.txt", io.BytesIO("Việt vị: luật 11".encode()))   # new content => new ETag
        assert sync_documents()["sport"] == 1
        assert "luật 11" in sandbox["backend"].chunks[("sport", "laws.txt")][0].page_content

        s3.delete(key)
        assert sync_documents()["removed"] == 1 and list_documents() == []
