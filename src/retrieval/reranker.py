"""
Reranking Module - Reranks retrieved documents by relevance
Uses CrossEncoder models for better relevance scoring
"""

from typing import List, Optional, Tuple
import logging

import os

import torch
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from config.settings import settings
from src.retrieval.lang import pick_query

logger = logging.getLogger(__name__)

torch.set_num_threads(settings.torch_threads or os.cpu_count() or 1)


class Reranker:
    """Reranks documents using CrossEncoder models"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize reranker with a CrossEncoder model
        
        Args:
            model_name: Model name from HuggingFace (defaults to settings.reranker_model)
        """
        if model_name is None:
            model_name = settings.reranker_model_path

        self.model_name = model_name
        self.model: Optional[CrossEncoder] = None
        # Pre-load the model immediately so first query doesn't pay the cold-start cost
        self._load_model()

    def _load_model(self) -> bool:
        if self.model is not None:
            return True

        try:
            logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(
                self.model_name,
                max_length=settings.reranker_max_length,
                device=settings.embedding_device,
            )
            logger.info("Reranker model loaded and ready.")
            return True
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            return False
    
    def rerank(self, query: str, documents: List[Document], k: int = None,
               query_variants: Optional[List[str]] = None) -> List[Tuple[Document, float]]:
        """
        Rerank documents by relevance to query

        Args:
            query: Query string
            documents: List of documents to rerank
            k: Return top k documents (uses top_k_rerank from settings if not specified)
            query_variants: Other renderings of the query (e.g. a translation). Each document
                is then scored against the variant in its own language (cross-lingual fix);
                the number of model calls stays one per document.

        Returns:
            List of (document, score) tuples sorted by relevance score descending
        """
        if k is None:
            k = settings.top_k_rerank
        
        if not documents:
            logger.warning("No documents to rerank")
            return []

        if not self._load_model():
            return [(doc, 0.5) for doc in documents[:k]]
        
        try:
            # Prepare pairs for CrossEncoder
            doc_texts = [doc.page_content for doc in documents]
            variants = [query] + [q for q in (query_variants or []) if q and q != query]
            if len(variants) > 1:
                pairs = [[pick_query(variants, text), text] for text in doc_texts]
            else:
                pairs = [[query, text] for text in doc_texts]

            # Get relevance scores
            logger.info(f"Reranking {len(documents)} documents for query")
            scores = self.model.predict(
                pairs, batch_size=settings.reranker_batch_size, show_progress_bar=False
            )
            
            # Sort by score descending
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs = sorted(((d, float(sc)) for d, sc in doc_score_pairs), key=lambda x: x[1], reverse=True)
            
            # Return top k
            top_k_results = doc_score_pairs[:k]
            logger.info(f"Reranking complete. Top-{k} selected.")
            
            return top_k_results
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fallback: return documents as-is with equal scores
            return [(doc, 0.5) for doc in documents[:k]]
    
    def batch_rerank(self, 
                     query: str, 
                     document_batches: dict, 
                     k: int = None) -> dict:
        """
        Rerank documents from multiple domains
        
        Args:
            query: Query string
            document_batches: Dict with domain names as keys and document lists as values
            k: Return top k per domain
        
        Returns:
            Dict with reranked documents per domain
        """
        if k is None:
            k = settings.top_k_rerank
        
        results = {}
        for domain, documents in document_batches.items():
            logger.info(f"Reranking {len(documents)} documents from {domain} domain")
            reranked = self.rerank(query, documents, k)
            results[domain] = reranked
        
        return results
    
    def score_documents(self, query: str, documents: List[Document]) -> List[dict]:
        """
        Score documents without reranking
        
        Args:
            query: Query string
            documents: List of documents
        
        Returns:
            List of dicts with document and score
        """
        if not self._load_model():
            return [
                {
                    "document": doc,
                    "score": 0.5,
                    "content_preview": doc.page_content[:100] + "..."
                }
                for doc in documents
            ]

        try:
            doc_texts = [doc.page_content for doc in documents]
            pairs = [[query, text] for text in doc_texts]
            scores = self.model.predict(
                pairs, batch_size=settings.reranker_batch_size, show_progress_bar=False
            )
            
            results = [
                {
                    "document": doc,
                    "score": float(score),
                    "content_preview": doc.page_content[:100] + "..."
                }
                for doc, score in zip(documents, scores)
            ]
            
            return sorted(results, key=lambda x: x["score"], reverse=True)
        except Exception as e:
            logger.error(f"Error scoring documents: {e}")
            return []


# Singleton instance
reranker = Reranker()
