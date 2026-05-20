"""
RAG Agent - Specialist agent for retrieving and generating answers from domain-specific vector stores
"""

from typing import Any, Dict, List, Optional
import logging
import json

from langchain_core.documents import Document

from .base_agent import BaseAgent
from src.retrieval import vector_store_manager, reranker, query_expander
from config.settings import settings

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """Specialist RAG agent for a specific domain"""
    
    def __init__(self, domain: str, name: str = None, model: str = None, temperature: float = None):
        """
        Initialize RAG agent
        
        Args:
            domain: Domain name (finance, tech, general)
            name: Agent name (defaults to "{domain} RAG Agent")
            model: LLM model
            temperature: Temperature
        """
        self.domain = domain
        self.name = name or f"{domain.capitalize()} RAG Agent"
        
        super().__init__(name=self.name, model=model, temperature=temperature)
        
        self.retriever = vector_store_manager.get_retriever(domain)
        if self.retriever is None:
            logger.warning(f"No vector store available for domain: {domain}")
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run RAG pipeline: retrieve, rerank, generate
        
        Args:
            input_data: Must contain 'query' key
        
        Returns:
            Output dict with answer, sources, metadata
        """
        try:
            query = input_data.get("query", "")
            if not query:
                return {
                    "success": False,
                    "answer": "No query provided",
                    "sources": [],
                    "domain": self.domain
                }
            
            self.add_to_history("user", query)
            
            # Step 1: Retrieve documents
            retrieved_docs = self._retrieve(query)
            
            if not retrieved_docs:
                return {
                    "success": False,
                    "answer": f"No relevant documents found in {self.domain} domain",
                    "sources": [],
                    "domain": self.domain,
                    "retrieved_count": 0
                }
            
            # Step 2: Rerank documents
            reranked_docs = self._rerank(query, retrieved_docs)
            
            # Step 3: Generate answer
            answer, citations = self._generate(query, reranked_docs)
            
            self.add_to_history("assistant", answer)
            
            return {
                "success": True,
                "answer": answer,
                "sources": citations,
                "domain": self.domain,
                "retrieved_count": len(retrieved_docs),
                "reranked_count": len(reranked_docs),
                "model": self.model
            }
        
        except Exception as e:
            logger.error(f"Error in RAG agent run: {e}")
            return {
                "success": False,
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "domain": self.domain,
                "error": str(e)
            }
    
    def _retrieve(self, query: str) -> List[Document]:
        """
        Retrieve documents from vector store
        
        Args:
            query: Search query
        
        Returns:
            List of retrieved documents
        """
        try:
            # Query expansion for better retrieval
            expanded_queries = query_expander.get_expanded_queries(
                query, 
                domain=self.domain,
                expansion_types=["multi_query", "synonyms"]
            )
            
            all_docs = []
            seen_content = set()
            
            for expanded_query in expanded_queries:
                docs = vector_store_manager.search(
                    domain=self.domain,
                    query=expanded_query,
                    k=settings.top_k_retrieval
                )
                
                # Deduplicate
                for doc in docs:
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_docs.append(doc)
            
            logger.info(f"Retrieved {len(all_docs)} documents from {self.domain}")
            return all_docs[:settings.top_k_retrieval * 2]  # Limit total
        
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def _rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents by relevance
        
        Args:
            query: Search query
            documents: Documents to rerank
        
        Returns:
            Reranked documents
        """
        if not settings.use_reranker or not documents:
            return documents
        
        try:
            reranked = reranker.rerank(query, documents, k=settings.top_k_rerank)
            logger.info(f"Reranked documents: {len(documents)} -> {len(reranked)}")
            return [doc for doc, score in reranked]
        
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return documents[:settings.top_k_rerank]
    
    def _generate(self, query: str, documents: List[Document]) -> tuple:
        """
        Generate answer from documents
        
        Args:
            query: Original query
            documents: Relevant documents
        
        Returns:
            Tuple of (answer, citations)
        """
        try:
            # Format context
            context_str = self._format_context_for_llm(documents)
            
            system_prompt = f"""You are a helpful assistant specializing in {self.domain} domain.
Answer questions based on the provided context.
If the context doesn't contain relevant information, say so clearly.
Always cite your sources using [Source N] notation.
Be precise and accurate."""
            
            prompt = f"""Context:
{context_str}

Question: {query}

Please provide a comprehensive answer with citations:"""
            
            answer = self.invoke([prompt], system_prompt=system_prompt)
            
            # Extract citations
            citations = self._extract_citations(documents, answer)
            
            return answer, citations
        
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return f"Error generating answer: {str(e)}", []
    
    def _format_context_for_llm(self, documents: List[Document]) -> str:
        """Format documents as context for LLM"""
        context = ""
        for i, doc in enumerate(documents, 1):
            content = doc.page_content[:500]  # Limit length
            source = doc.metadata.get("source", "Unknown")
            context += f"\n[Source {i}] {source}\n{content}\n"
        return context
    
    def _extract_citations(self, documents: List[Document], answer: str) -> List[Dict]:
        """Extract citations from answer"""
        citations = []
        
        for i, doc in enumerate(documents, 1):
            if f"[Source {i}]" in answer:
                citations.append({
                    "source_id": i,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", None)
                })
        
        return citations if citations else []
    
    def batch_process(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple queries
        
        Args:
            queries: List of queries
        
        Returns:
            List of results
        """
        results = []
        for query in queries:
            result = self.run({"query": query})
            results.append(result)
        return results
    
    def get_domain_stats(self) -> Dict[str, Any]:
        """Get statistics about this agent's domain"""
        return {
            "agent": self.name,
            "domain": self.domain,
            "store_stats": vector_store_manager.get_store_stats(self.domain),
            "model": self.model,
            "conversation_history_length": len(self.conversation_history)
        }
