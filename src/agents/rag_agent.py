"""
RAG Agent - Specialist agent that retrieves from a domain vector store and
generates a cited answer.

Pipeline: query expansion -> hybrid retrieval -> cross-encoder rerank ->
relevance threshold -> (cross-domain fallback -> web fallback) -> generation.
"""

from typing import Any, Dict, List, Tuple
import logging

from langchain_core.documents import Document

from .base_agent import BaseAgent
from src.retrieval import reranker, query_expander
from src.retrieval.vector_store import get_vector_store_manager  # resolved on use: importing agents loads no model
from src.tools.web_search import web_search_tool
from config.settings import settings
from src.guardrails.injection import DATA_RULE, neutralise
from src.observability import metrics

logger = logging.getLogger(__name__)

ALL_DOMAINS = ["sport", "tech", "general"]
MAX_CONTEXT_CHARS = 1500

ScoredDoc = Tuple[Document, float]


class RAGAgent(BaseAgent):
    """Specialist RAG agent for a specific domain"""

    def __init__(self, domain: str, name: str = None, model: str = None, temperature: float = None):
        self.domain = domain
        name = name or f"{domain.capitalize()} RAG Agent"
        super().__init__(name=name, model=model, temperature=temperature)

    # ------------------------------------------------------------------ main

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            input_data: Must contain 'query'
        """
        try:
            query = input_data.get("query", "")
            if not query:
                return {"success": False, "answer": "No query provided", "sources": [], "domain": self.domain}

            documents, retrieved_count, fallback = self._gather_context(
                query, input_data.get("expanded_queries") or None
            )
            answer, citations = self._generate(
                query, documents, input_data.get("on_token"), input_data.get("history", "")
            )

            return {
                "success": True,
                "answer": answer,
                "sources": citations,
                "contexts": self._contexts(documents),
                "domain": self.domain,
                "fallback": fallback,
                "retrieved_count": retrieved_count,
                "reranked_count": len(documents),
                "model": self.model,
            }

        except Exception as e:
            logger.error(f"Error in RAG agent run: {e}", exc_info=True)
            return {
                "success": False,
                "answer": f"Error processing query: {e}",
                "sources": [],
                "domain": self.domain,
                "error": str(e),
            }

    def _gather_context(self, query: str, expanded: List[str] = None) -> Tuple[List[Document], int, str]:
        """
        Return (documents, retrieved_count, fallback_used).

        A chunk only counts as relevant if its rerank score reaches
        settings.relevance_threshold; plain similarity search always returns k
        neighbours, relevant or not, so without this the fallbacks never fire.
        """
        # 1) Own domain (with query expansion; the rewrites include a vi<->en translation)
        candidates, variants = self._retrieve(query, self.domain, expand=True, expanded=expanded)
        retrieved_count = len(candidates)
        relevant = self._select_relevant(query, candidates, variants)
        if relevant:
            return relevant, retrieved_count, "none"

        # 2) Other local domains
        logger.warning(f"No relevant docs in '{self.domain}', trying other domains...")
        other: List[Document] = []
        for domain in ALL_DOMAINS:
            if domain != self.domain:
                other.extend(self._retrieve(query, domain, expand=False)[0])
        retrieved_count += len(other)
        relevant = self._select_relevant(query, other, variants)
        if relevant:
            return relevant, retrieved_count, "cross_domain"

        # 3) Live web
        logger.warning(f"No relevant local docs for '{query}', falling back to web search...")
        web_docs = self._web_documents(query)
        retrieved_count += len(web_docs)
        if web_docs:
            return web_docs, retrieved_count, "web"

        return [], retrieved_count, "none"

    # ------------------------------------------------------------ retrieval

    def _retrieve(self, query: str, domain: str, expand: bool,
                  expanded: List[str] = None) -> Tuple[List[Document], List[str]]:
        """Hybrid-search a domain (optionally for several expanded queries) and dedupe.

        Returns (candidates, query variants used)."""
        try:
            queries = [query]
            if expand and expanded and expanded[0] == query:
                queries = list(expanded)  # already expanded while routing
            elif expand:
                queries = query_expander.get_expanded_queries(
                    query, domain=domain, expansion_types=["multi_query"]
                )

            per_query = [
                get_vector_store_manager().hybrid_search(domain, q, k=settings.top_k_retrieval) for q in queries
            ]

            # Interleave the per-query rankings (best of each query first), dedupe, then cap:
            # the reranker is the slowest step and costs time per candidate.
            docs: List[Document] = []
            seen = set()
            for rank in range(max((len(r) for r in per_query), default=0)):
                for results in per_query:
                    if rank < len(results):
                        doc = results[rank]
                        key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content)
                        if key not in seen:
                            seen.add(key)
                            docs.append(doc)

            logger.info(f"Retrieved {len(docs)} candidate documents from {domain}")
            return docs[: settings.rerank_max_candidates], queries

        except Exception as e:
            logger.error(f"Error during retrieval in {domain}: {e}", exc_info=True)
            return [], [query]

    def _select_relevant(self, query: str, documents: List[Document],
                         variants: List[str] = None) -> List[Document]:
        """Rerank and keep only chunks above the relevance threshold."""
        if not documents:
            return []

        if not settings.use_reranker:
            return documents[: settings.top_k_rerank]

        try:
            scored: List[ScoredDoc] = reranker.rerank(
                query, documents, k=settings.top_k_rerank, query_variants=variants
            )
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return documents[: settings.top_k_rerank]

        relevant = [doc for doc, score in scored if score >= settings.relevance_threshold]
        best = max((float(s) for _, s in scored), default=float("nan"))
        logger.info(
            f"Rerank: {len(documents)} candidates -> {len(relevant)} relevant "
            f"(best score {best:.2f}, threshold {settings.relevance_threshold})"
        )
        return relevant

    def _web_documents(self, query: str) -> List[Document]:
        try:
            results = web_search_tool.search(query, limit=settings.top_k_rerank + 1)
        except Exception as e:
            logger.error(f"Error in web search fallback: {e}")
            return []

        return [
            Document(
                page_content=f"{item['title']}: {item['snippet']}",
                metadata={
                    "source": item.get("url", ""),
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "domain": "web",
                    "type": "web_search",
                },
            )
            for item in results
        ]

    # ----------------------------------------------------------- generation

    def _generate(self, query: str, documents: List[Document], on_token=None, history: str = "") -> Tuple[str, List[Dict]]:
        """Generate an answer from documents"""
        try:
            system_prompt = f"""You are a helpful assistant specializing in the {self.domain} domain.
Answer the question based on the provided context, in the same language as the question.
If the context is empty or insufficient, answer from your own knowledge but clearly say so.
Always cite sources using [Source N] notation when context is available.
Be precise and accurate.
{DATA_RULE}"""

            if documents:
                prompt = f"""Context:
{self._format_context_for_llm(documents)}

Question: {query}

Please provide a comprehensive answer with citations:"""
            else:
                prompt = (
                    f"Question: {query}\n\n"
                    "No relevant documents were found in the knowledge base or on the web. "
                    "Answer from your general knowledge and note this clearly."
                )

            if history:
                prompt = (
                    f"Earlier conversation (for context only; do not cite it):\n{history}\n\n" + prompt
                )
            answer = self.invoke([prompt], system_prompt=system_prompt, on_token=on_token)
            return answer, self._extract_citations(documents, answer)

        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            raise

    def _format_context_for_llm(self, documents: List[Document]) -> str:
        context, removed = "", 0
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("title") or doc.metadata.get("filename") or doc.metadata.get("source", "Unknown")
            text, n = neutralise(doc.page_content[:MAX_CONTEXT_CHARS])   # indirect prompt injection
            removed += n
            context += f"\n[Source {i}] {source}\n{text}\n"
        if removed:
            logger.warning(f"Neutralised {removed} injection-like phrase(s) in {self.domain} context")
            metrics.record_injection(self.domain, removed)
        return context

    @staticmethod
    def _contexts(documents: List[Document]) -> List[Dict]:
        """Every chunk shown to the LLM, numbered as in the prompt (for evaluation)."""
        return [
            {
                "source_id": i,
                "source": doc.metadata.get("source", "Unknown"),
                "title": doc.metadata.get("title") or doc.metadata.get("filename"),
                "page": doc.metadata.get("page"),
                "url": doc.metadata.get("url"),
                "text": doc.page_content[:MAX_CONTEXT_CHARS],
            }
            for i, doc in enumerate(documents, 1)
        ]

    def _extract_citations(self, documents: List[Document], answer: str) -> List[Dict]:
        """Citations for the [Source N] markers actually used in the answer"""
        citations = []
        for i, doc in enumerate(documents, 1):
            if f"[Source {i}]" in answer:
                meta = doc.metadata
                citation = {
                    "source_id": i,
                    "source": meta.get("source", "Unknown"),
                    "title": meta.get("title") or meta.get("filename"),
                    "snippet": doc.page_content[:200],
                    "page": meta.get("page"),
                }
                if meta.get("url"):
                    citation["url"] = meta["url"]
                citations.append(citation)
        return citations

    # -------------------------------------------------------------- helpers

    def batch_process(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries"""
        return [self.run({"query": q}) for q in queries]

    def get_domain_stats(self) -> Dict[str, Any]:
        """Get statistics about this agent's domain"""
        return {
            "agent": self.name,
            "domain": self.domain,
            "store_stats": get_vector_store_manager().get_store_stats(self.domain),
            "model": self.model,
        }
