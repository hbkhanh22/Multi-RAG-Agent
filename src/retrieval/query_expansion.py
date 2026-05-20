"""
Query Expansion Module - Expands queries to improve retrieval coverage
Includes HyDE (Hypothetical Document Embeddings) and Multi-Query expansion
"""

from typing import List
import logging
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings

logger = logging.getLogger(__name__)


class QueryExpander:
    """Expands queries using various techniques to improve retrieval"""
    
    def __init__(self):
        """Initialize query expander with Google LLM"""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=settings.fast_llm_model,
                google_api_key=settings.google_api_key,
                temperature=0.3,
                max_tokens=1000
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _invoke_llm(self, messages: list) -> str:
        """Invoke the LLM and handle quota or transient errors."""
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            error_text = str(e)
            if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text or "quota" in error_text.lower():
                logger.warning(f"LLM quota exhausted during query expansion: {error_text}")
            else:
                logger.error(f"LLM error during query expansion: {error_text}")
            return ""
    
    def hyde_expansion(self, query: str) -> str:
        """
        Hypothetical Document Embeddings (HyDE) - Generate a hypothetical 
        document that would answer the query
        
        Args:
            query: Original query
        
        Returns:
            Hypothetical document
        """
        try:
            system_prompt = """You are an expert at generating hypothetical documents.
Given a query, generate a realistic document that would answer it perfectly.
Focus on the specific content and details that would be in such a document.
Generate only the document content, nothing else."""
            
            prompt = f"""Generate a hypothetical document that would answer this query:
{query}

Hypothetical document:"""
            
            content = self._invoke_llm([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])
            
            if not content:
                return query
            
            logger.info("HyDE expansion generated successfully")
            return content
        except Exception as e:
            logger.error(f"Error in HyDE expansion: {e}")
            return query  # Fallback to original query
    
    def multi_query_expansion(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple reformulations of the query to improve recall
        
        Args:
            query: Original query
            num_queries: Number of alternative queries to generate
        
        Returns:
            List of alternative queries
        """
        try:
            system_prompt = f"""You are an expert at reformulating queries for search.
Generate {num_queries} different phrasings of the given query that capture different aspects or terminology.
Each phrasing should be a valid search query.
Return ONLY a JSON array of strings, nothing else.

Example format:
["query 1", "query 2", "query 3"]"""
            
            prompt = f"""Original query: {query}

Generate {num_queries} alternative queries that capture different aspects:"""
            
            content = self._invoke_llm([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])
            
            if not content:
                return [query]
            
            try:
                alternatives = json.loads(content)
                if isinstance(alternatives, list):
                    expanded_queries = [query] + alternatives[:num_queries]
                    logger.info(f"Generated {len(alternatives)} alternative queries")
                    return expanded_queries
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response, returning original query")
                return [query]
        except Exception as e:
            logger.error(f"Error in multi-query expansion: {e}")
            return [query]
    
    def domain_specific_expansion(self, query: str, domain: str) -> List[str]:
        """
        Generate domain-specific query variations
        
        Args:
            query: Original query
            domain: Domain (finance, tech, general)
        
        Returns:
            List of domain-specific queries
        """
        try:
            domain_context = {
                "finance": "stock market, investment, economics, financial reports, trading",
                "tech": "programming, software, hardware, AI, cloud, DevOps",
                "general": "history, science, culture, news, education"
            }
            
            context = domain_context.get(domain, "general knowledge")
            
            system_prompt = f"""You are an expert in {domain} domain search optimization.
Generate 2 variations of the given query optimized for {domain} context.
Consider relevant terminology, concepts, and common phrasing in {domain}.
Return ONLY a JSON array of strings, nothing else.

Example format:
["query 1", "query 2"]"""
            
            prompt = f"""Original query: {query}

Domain: {domain} ({context})
Generate 2 domain-optimized query variations:"""
            
            content = self._invoke_llm([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])
            
            if not content:
                return [query]
            
            try:
                variations = json.loads(content)
                if isinstance(variations, list):
                    return [query] + variations[:2]
            except json.JSONDecodeError:
                return [query]
        except Exception as e:
            logger.error(f"Error in domain-specific expansion: {e}")
            return [query]
    
    def expand_with_synonyms(self, query: str) -> List[str]:
        """
        Generate synonymous query formulations
        
        Args:
            query: Original query
        
        Returns:
            List of synonymous queries
        """
        try:
            system_prompt = """Generate 2 synonymous versions of the given query using different vocabulary.
Return ONLY a JSON array of strings, nothing else.

Example format:
["query 1", "query 2"]"""
            
            prompt = f"""Query: {query}

Generate 2 synonymous query formulations with different vocabulary:"""
            
            content = self._invoke_llm([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])
            
            if not content:
                return [query]
            
            try:
                synonyms = json.loads(content)
                if isinstance(synonyms, list):
                    return [query] + synonyms[:2]
            except json.JSONDecodeError:
                return [query]
        except Exception as e:
            logger.error(f"Error in synonym expansion: {e}")
            return [query]
    
    def get_expanded_queries(self, query: str, domain: str = None, 
                            expansion_types: List[str] = None) -> List[str]:
        """
        Get expanded queries using multiple techniques
        
        Args:
            query: Original query
            domain: Optional domain for domain-specific expansion
            expansion_types: List of expansion types to use
                           (hyde, multi_query, domain_specific, synonyms)
        
        Returns:
            Combined list of original and expanded queries
        """
        if expansion_types is None:
            expansion_types = ["multi_query", "synonyms"]
        
        expanded = [query]  # Start with original
        
        try:
            for expansion_type in expansion_types:
                if expansion_type == "hyde":
                    hyde_doc = self.hyde_expansion(query)
                    expanded.append(hyde_doc)
                
                elif expansion_type == "multi_query":
                    multi_queries = self.multi_query_expansion(query)
                    expanded.extend(multi_queries[1:])  # Skip original
                
                elif expansion_type == "domain_specific" and domain:
                    domain_queries = self.domain_specific_expansion(query, domain)
                    expanded.extend(domain_queries[1:])  # Skip original
                
                elif expansion_type == "synonyms":
                    synonym_queries = self.expand_with_synonyms(query)
                    expanded.extend(synonym_queries[1:])  # Skip original
            
            # Remove duplicates while preserving order
            seen = set()
            unique_expanded = []
            for q in expanded:
                if q not in seen:
                    seen.add(q)
                    unique_expanded.append(q)
            
            logger.info(f"Generated {len(unique_expanded)} total queries from expansion")
            return unique_expanded
        
        except Exception as e:
            logger.error(f"Error in get_expanded_queries: {e}")
            return [query]


# Singleton instance
query_expander = QueryExpander()
