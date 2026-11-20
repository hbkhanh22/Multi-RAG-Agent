"""
Vector Store Module - vector, BM25 and hybrid (RRF-fused) search per domain.

Storage is pluggable (settings.vector_backend):
  chroma    embedded Chroma collections under data/vectorstores (dev, single process)
  pgvector  one `chunks` table in PostgreSQL with an HNSW cosine index - shares the
            database with conversations / document registry, so a file's chunks are
            replaced in a single transaction and several API workers + the ingest worker
            see the same data.

The BM25 index is built in memory from the stored chunks and rebuilt when the
backend's fingerprint (row count / newest row) changes - including changes made by
another process such as the ingestion worker.
"""

import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config.settings import settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
RRF_K = 60  # standard Reciprocal Rank Fusion constant
FINGERPRINT_TTL = 5.0  # seconds between staleness checks of the BM25 index


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _doc_key(doc: Document) -> Tuple:
    return (
        doc.metadata.get("source"),
        doc.metadata.get("page"),
        doc.metadata.get("start_index"),
        doc.page_content[:80],
    )


def embedding_slug() -> str:
    """Tag of the embedding model + runtime + index version (store dir / index name)."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", settings.embedding_model.split("/")[-1]).lower()
    if settings.effective_model_backend == "onnx":
        slug += "-onnx-int8"   # int8 vectors are not interchangeable with fp32 ones
    version = re.sub(r"[^A-Za-z0-9._-]+", "_", settings.index_version or "").lower()
    return f"{slug}-{version}" if version else slug


def make_embeddings():
    """Embedding model for the configured runtime (torch fp32 or ONNX int8)."""
    if settings.effective_model_backend == "onnx":
        from src.retrieval.onnx_models import OnnxEmbeddings, onnx_available

        if onnx_available(settings.embedding_model):
            return OnnxEmbeddings()
        raise RuntimeError(
            f"MODEL_BACKEND=onnx but no ONNX model in {settings.onnx_model_dir}; "
            "run `python scripts/export_onnx.py` first"
        )
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_path,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": settings.embedding_batch_size},
    )


# ====================================================================== backends

class VectorBackend(ABC):
    """Chunk storage + nearest-neighbour search for one embedding index."""

    name = "base"

    @abstractmethod
    def add(self, domain: str, docs: List[Document]) -> int: ...

    @abstractmethod
    def replace_file(self, domain: str, filename: str, docs: List[Document]) -> int:
        """Atomically (where supported) swap all chunks of `filename` for `docs`."""

    @abstractmethod
    def delete_file(self, domain: str, filename: str) -> int: ...

    @abstractmethod
    def search(self, domain: str, query: str, k: int) -> List[Document]: ...

    @abstractmethod
    def all_documents(self, domain: str) -> List[Document]: ...

    @abstractmethod
    def count(self, domain: str) -> int: ...

    def fingerprint(self, domain: str):
        return self.count(domain)

    def healthy(self) -> bool:
        return True


class ChromaBackend(VectorBackend):
    name = "chroma"

    def __init__(self, embeddings, index_name: str):
        self.embeddings = embeddings
        self.index_name = index_name
        self.stores: Dict[str, object] = {}
        for domain in VectorStoreManager.DOMAINS:
            try:
                self.stores[domain] = self._make(domain)
            except Exception as e:
                logger.error(f"Error loading/creating {domain} Chroma store: {e}")
                self.stores[domain] = None

    def _path(self, domain: str) -> Path:
        base = {
            "sport": settings.sport_store_path,
            "tech": settings.tech_store_path,
            "general": settings.general_store_path,
        }[domain]
        return Path(base) / self.index_name

    def _make(self, domain: str):
        from langchain_chroma import Chroma

        path = self._path(domain)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Loading {domain} Chroma store from {path}")
        return Chroma(persist_directory=str(path), embedding_function=self.embeddings,
                      collection_name=f"{domain}_collection")

    def _store(self, domain: str):
        store = self.stores.get(domain)
        if store is None:
            raise RuntimeError(f"Store for domain '{domain}' not available")
        return store

    def add(self, domain, docs):
        if docs:
            self._store(domain).add_documents(docs)
        return len(docs)

    def replace_file(self, domain, filename, docs):
        self.delete_file(domain, filename)
        return self.add(domain, docs)

    def delete_file(self, domain, filename):
        store = self._store(domain)
        ids = store._collection.get(where={"filename": filename}).get("ids", [])
        if ids:
            store.delete(ids=ids)
        return len(ids)

    def search(self, domain, query, k):
        return self._store(domain).similarity_search(query, k=k)

    def all_documents(self, domain):
        stored = self._store(domain).get(include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(stored.get("documents") or [], stored.get("metadatas") or [])
            if text
        ]

    def count(self, domain):
        return self._store(domain)._collection.count()



class PgVectorBackend(VectorBackend):
    name = "pgvector"
    INSERT_BATCH = 64

    def __init__(self, embeddings, index_name: str, url: Optional[str] = None):
        from src.storage.db import get_engine

        self.embeddings = embeddings
        self.index_name = index_name
        self.engine = get_engine(url)
        if self.engine.dialect.name != "postgresql":
            raise RuntimeError("VECTOR_BACKEND=pgvector needs DATABASE_URL=postgresql+psycopg://...")

    def _where(self, domain: str, filename: Optional[str] = None):
        from src.storage.db import chunks

        cond = [chunks.c.index_name == self.index_name, chunks.c.domain == domain]
        if filename is not None:
            cond.append(chunks.c.filename == filename)
        return cond

    def _rows(self, domain: str, docs: List[Document]) -> List[dict]:
        vectors = self.embeddings.embed_documents([d.page_content for d in docs])
        return [
            {
                "index_name": self.index_name,
                "domain": domain,
                "source_key": str(d.metadata.get("source", "")),
                "filename": str(d.metadata.get("filename", "")),
                "content": d.page_content.replace("\x00", ""),   # Postgres TEXT rejects NUL
                "meta": {k: v for k, v in d.metadata.items() if isinstance(v, (str, int, float, bool)) or v is None},
                "embedding": vec,
            }
            for d, vec in zip(docs, vectors)
        ]

    def _insert(self, conn, rows: List[dict]) -> None:
        from sqlalchemy import insert
        from src.storage.db import chunks

        for i in range(0, len(rows), self.INSERT_BATCH):
            conn.execute(insert(chunks), rows[i:i + self.INSERT_BATCH])

    def add(self, domain, docs):
        if not docs:
            return 0
        rows = self._rows(domain, docs)   # embed outside the transaction
        with self.engine.begin() as conn:
            self._insert(conn, rows)
        return len(rows)

    def replace_file(self, domain, filename, docs):
        from sqlalchemy import delete
        from src.storage.db import chunks

        rows = self._rows(domain, docs) if docs else []
        with self.engine.begin() as conn:   # readers see either the old or the new version
            conn.execute(delete(chunks).where(*self._where(domain, filename)))
            self._insert(conn, rows)
        return len(rows)

    def delete_file(self, domain, filename):
        from sqlalchemy import delete
        from src.storage.db import chunks

        with self.engine.begin() as conn:
            return conn.execute(delete(chunks).where(*self._where(domain, filename))).rowcount

    def search(self, domain, query, k):
        from sqlalchemy import select, text
        from src.storage.db import chunks

        qv = self.embeddings.embed_query(query)
        distance = chunks.c.embedding.cosine_distance(qv).label("distance")
        with self.engine.begin() as conn:
            # pgvector >= 0.8: keep scanning the HNSW graph until k rows pass the
            # domain filter (plain HNSW + WHERE can return fewer than k rows)
            conn.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            conn.execute(text("SET LOCAL hnsw.ef_search = 100"))
            rows = conn.execute(
                select(chunks.c.content, chunks.c.meta, distance)
                .where(*self._where(domain)).order_by(distance).limit(k)
            ).all()
        rows = sorted(rows, key=lambda r: r.distance)   # relaxed_order => re-sort
        return [Document(page_content=r.content, metadata=dict(r.meta or {})) for r in rows]

    def all_documents(self, domain):
        from sqlalchemy import select
        from src.storage.db import chunks

        with self.engine.connect() as conn:
            rows = conn.execute(
                select(chunks.c.content, chunks.c.meta).where(*self._where(domain)).order_by(chunks.c.id)
            ).all()
        return [Document(page_content=r.content, metadata=dict(r.meta or {})) for r in rows]

    def count(self, domain):
        from sqlalchemy import func, select
        from src.storage.db import chunks

        with self.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(chunks).where(*self._where(domain))).scalar_one()

    def fingerprint(self, domain):
        from sqlalchemy import func, select
        from src.storage.db import chunks

        with self.engine.connect() as conn:
            row = conn.execute(
                select(func.count(), func.coalesce(func.max(chunks.c.id), 0)).where(*self._where(domain))
            ).one()
        return tuple(row)


    def healthy(self):
        from src.storage.db import ping
        return ping(self.engine)


# ====================================================================== manager

class VectorStoreManager:
    """Domain-aware facade used by the agents and the ingestion pipeline."""

    DOMAINS = ("sport", "tech", "general")

    def __init__(self, backend: Optional[VectorBackend] = None, embeddings=None):
        try:
            self.embeddings = embeddings or (backend.embeddings if backend else make_embeddings())
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
        self.index_name = embedding_slug()
        self.backend = backend or self._make_backend()
        self._bm25_cache: Dict[str, Tuple[object, Optional[Tuple[BM25Okapi, List[Document]]]]] = {}
        self._bm25_checked: Dict[str, float] = {}
        self._lock = threading.Lock()

    def _make_backend(self) -> VectorBackend:
        kind = settings.vector_backend.lower()
        logger.info(f"Vector backend: {kind} (index '{self.index_name}')")
        if kind == "pgvector":
            return PgVectorBackend(self.embeddings, self.index_name)
        if kind == "chroma":
            return ChromaBackend(self.embeddings, self.index_name)
        raise ValueError(f"Unknown VECTOR_BACKEND '{settings.vector_backend}' (chroma|pgvector)")

    def _invalidate_bm25(self, domain: str):
        with self._lock:
            self._bm25_cache.pop(domain, None)
            self._bm25_checked.pop(domain, None)

    # ------------------------------------------------------------------ write

    def add_documents(self, domain: str, documents: List[Document]) -> bool:
        try:
            logger.info(f"Adding {len(documents)} documents to {domain} store")
            self.backend.add(domain, documents)
            self._invalidate_bm25(domain)
            return True
        except Exception as e:
            logger.error(f"Error adding documents to {domain} store: {e}")
            return False

    def replace_file(self, domain: str, filename: str, documents: List[Document]) -> int:
        """Replace every chunk of `filename` (raises on failure)."""
        n = self.backend.replace_file(domain, filename, documents)
        self._invalidate_bm25(domain)
        return n

    def delete_by_filename(self, domain: str, filename: str) -> int:
        """Delete every chunk that came from `filename`. Returns number removed."""
        try:
            n = self.backend.delete_file(domain, filename)
            if n:
                self._invalidate_bm25(domain)
            return n
        except Exception as e:
            logger.error(f"Error deleting '{filename}' from {domain}: {e}")
            return 0

    # ------------------------------------------------------------------ read

    def search(self, domain: str, query: str, k: int = None) -> List[Document]:
        """Plain vector similarity search (returns the k nearest chunks)"""
        k = k or settings.top_k_retrieval
        try:
            docs = self.backend.search(domain, query, k)
            logger.info(f"Vector search: {len(docs)} documents in {domain}")
            return docs
        except Exception as e:
            logger.error(f"Error searching {domain} store: {e}")
            return []

    def _bm25_index(self, domain: str) -> Optional[Tuple[BM25Okapi, List[Document]]]:
        now = time.monotonic()
        with self._lock:
            cached = self._bm25_cache.get(domain)
            if cached is not None and now - self._bm25_checked.get(domain, 0) < FINGERPRINT_TTL:
                return cached[1]

        fp = self.backend.fingerprint(domain)
        if cached is not None and cached[0] == fp:
            with self._lock:
                self._bm25_checked[domain] = now
            return cached[1]

        docs = self.backend.all_documents(domain)
        index = (BM25Okapi([_tokenize(d.page_content) for d in docs]), docs) if docs else None
        with self._lock:
            self._bm25_cache[domain] = (fp, index)
            self._bm25_checked[domain] = now
        return index

    def bm25_search(self, domain: str, query: str, k: int = None) -> List[Document]:
        """Keyword (BM25) search over a domain's chunks"""
        k = k or settings.top_k_retrieval
        index = self._bm25_index(domain)
        tokens = _tokenize(query)
        if index is None or not tokens:
            return []

        bm25, docs = index
        scores = bm25.get_scores(tokens)
        ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]
        return [docs[i] for i in ranked if scores[i] > 0]

    def hybrid_search(self, domain: str, query: str, k: int = None) -> List[Document]:
        """Vector + BM25 search fused with weighted Reciprocal Rank Fusion"""
        k = k or settings.top_k_retrieval
        vector_docs = self.search(domain, query, k=k)
        try:
            bm25_docs = self.bm25_search(domain, query, k=k)
        except Exception as e:
            logger.error(f"BM25 search failed in {domain}: {e}")
            bm25_docs = []

        scores: Dict[Tuple, float] = {}
        docs_by_key: Dict[Tuple, Document] = {}
        for weight, ranked in ((settings.vector_weight, vector_docs), (settings.bm25_weight, bm25_docs)):
            for rank, doc in enumerate(ranked):
                key = _doc_key(doc)
                docs_by_key.setdefault(key, doc)
                scores[key] = scores.get(key, 0.0) + weight / (RRF_K + rank + 1)

        fused = sorted(scores, key=scores.get, reverse=True)[:k]
        return [docs_by_key[key] for key in fused]

    def get_store_stats(self, domain: str) -> dict:
        try:
            return {
                "domain": domain,
                "document_count": self.backend.count(domain),
                "embedding_dimension": settings.embedding_dimension,
                "embedding_model": settings.embedding_model,
                "backend": self.backend.name,
                "index": self.index_name,
                "status": "active",
            }
        except Exception as e:
            logger.error(f"Error getting stats for {domain}: {e}")
            return {"error": str(e)}

    def healthy(self) -> bool:
        try:
            return self.backend.healthy()
        except Exception:
            return False


# Singleton, created on first access (PEP 562) so that importing this module (worker,
# migrations, tests with a fake backend) does not load the embedding model.
_manager: Optional[VectorStoreManager] = None
_manager_lock = threading.Lock()


def get_vector_store_manager() -> VectorStoreManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = VectorStoreManager()
    return _manager


def __getattr__(name):
    if name == "vector_store_manager":
        return get_vector_store_manager()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
