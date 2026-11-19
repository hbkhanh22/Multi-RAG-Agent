"""
Vector Store Module - Manages one Chroma collection per domain and provides
vector, BM25 and hybrid (RRF-fused) search.
"""

import logging
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config.settings import settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
RRF_K = 60  # standard Reciprocal Rank Fusion constant


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
    """Filesystem-safe tag of the configured embedding model."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", settings.embedding_model.split("/")[-1]).lower()


class VectorStoreManager:
    """Manages vector stores for different domains"""

    DOMAINS = ("sport", "tech", "general")

    def __init__(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_path,
                model_kwargs={"device": settings.embedding_device},
                encode_kwargs={"normalize_embeddings": True, "batch_size": settings.embedding_batch_size},
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise

        self.stores: Dict[str, Optional[Chroma]] = {}
        self._bm25_cache: Dict[str, Optional[Tuple[BM25Okapi, List[Document]]]] = {}
        self._lock = threading.Lock()
        self._load_or_create_stores()

    # ------------------------------------------------------------------ setup

    def _store_path(self, domain: str) -> Path:
        base = {
            "sport": settings.sport_store_path,
            "tech": settings.tech_store_path,
            "general": settings.general_store_path,
        }[domain]
        # One sub-directory per embedding model: switching models never mixes
        # incompatible vectors and triggers a clean re-ingestion.
        return Path(base) / embedding_slug()

    def _make_store(self, domain: str) -> Chroma:
        path = self._store_path(domain)
        Path(path).mkdir(parents=True, exist_ok=True)
        return Chroma(
            persist_directory=str(path),
            embedding_function=self.embeddings,
            collection_name=f"{domain}_collection",
        )

    def _load_or_create_stores(self):
        for domain in self.DOMAINS:
            try:
                logger.info(f"Loading {domain} vector store from {self._store_path(domain)}")
                self.stores[domain] = self._make_store(domain)
            except Exception as e:
                logger.error(f"Error loading/creating {domain} store: {e}")
                self.stores[domain] = None

    def _get_store(self, domain: str) -> Optional[Chroma]:
        store = self.stores.get(domain)
        if store is None:
            logger.error(f"Store for domain '{domain}' not available")
        return store

    def _invalidate_bm25(self, domain: str):
        with self._lock:
            self._bm25_cache.pop(domain, None)

    # ------------------------------------------------------------------ write

    def add_documents(self, domain: str, documents: List[Document]) -> bool:
        """Add documents to a domain's vector store"""
        try:
            store = self._get_store(domain)
            if store is None:
                return False

            logger.info(f"Adding {len(documents)} documents to {domain} store")
            store.add_documents(documents)  # langchain_chroma persists automatically
            self._invalidate_bm25(domain)
            return True
        except Exception as e:
            logger.error(f"Error adding documents to {domain} store: {e}")
            return False

    def delete_by_filename(self, domain: str, filename: str) -> int:
        """Delete every chunk that came from `filename`. Returns number removed."""
        try:
            store = self._get_store(domain)
            if store is None:
                return 0
            ids = store._collection.get(where={"filename": filename}).get("ids", [])
            if ids:
                store.delete(ids=ids)
                self._invalidate_bm25(domain)
            return len(ids)
        except Exception as e:
            logger.error(f"Error deleting '{filename}' from {domain}: {e}")
            return 0

    def delete_documents(self, domain: str, document_ids: List[str]) -> bool:
        """Delete specific documents (by chunk id) from a store"""
        try:
            store = self._get_store(domain)
            if store is None:
                return False
            store.delete(ids=document_ids)
            self._invalidate_bm25(domain)
            return True
        except Exception as e:
            logger.error(f"Error deleting documents from {domain}: {e}")
            return False

    def clear_store(self, domain: str) -> bool:
        """Remove all documents from a store"""
        try:
            store = self._get_store(domain)
            if store is None:
                return False
            store.delete_collection()
            self.stores[domain] = self._make_store(domain)
            self._invalidate_bm25(domain)
            return True
        except Exception as e:
            logger.error(f"Error clearing {domain} store: {e}")
            return False

    # ------------------------------------------------------------------ read

    def get_retriever(self, domain: str, k: int = None, search_type: str = "mmr"):
        """Get a LangChain retriever for a domain"""
        k = k or settings.top_k_retrieval
        store = self._get_store(domain)
        if store is None:
            return None
        try:
            return store.as_retriever(search_type=search_type, search_kwargs={"k": k, "fetch_k": k * 2})
        except Exception as e:
            logger.error(f"Error creating retriever for {domain}: {e}")
            return None

    def search(self, domain: str, query: str, k: int = None) -> List[Document]:
        """Plain vector similarity search (returns the k nearest chunks)"""
        k = k or settings.top_k_retrieval
        try:
            store = self._get_store(domain)
            if store is None:
                return []
            docs = store.similarity_search(query, k=k)
            logger.info(f"Vector search: {len(docs)} documents in {domain}")
            return docs
        except Exception as e:
            logger.error(f"Error searching {domain} store: {e}")
            return []

    def _bm25_index(self, domain: str) -> Optional[Tuple[BM25Okapi, List[Document]]]:
        with self._lock:
            if domain in self._bm25_cache:
                return self._bm25_cache[domain]

        store = self._get_store(domain)
        index = None
        if store is not None:
            stored = store.get(include=["documents", "metadatas"])
            docs = [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(stored.get("documents") or [], stored.get("metadatas") or [])
                if text
            ]
            if docs:
                index = (BM25Okapi([_tokenize(d.page_content) for d in docs]), docs)

        with self._lock:
            self._bm25_cache[domain] = index
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
        """Get statistics about a vector store"""
        try:
            store = self.stores.get(domain)
            if store is None:
                return {"error": "Store not available"}

            return {
                "domain": domain,
                "document_count": store._collection.count(),
                "embedding_dimension": settings.embedding_dimension,
                "embedding_model": settings.embedding_model,
                "status": "active",
            }
        except Exception as e:
            logger.error(f"Error getting stats for {domain}: {e}")
            return {"error": str(e)}


# Singleton instance
vector_store_manager = VectorStoreManager()
