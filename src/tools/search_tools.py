"""
Search Tools - Web and knowledge base search capabilities for agents
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class SearchTool:
    """Base search tool"""
    
    def __init__(self, name: str):
        self.name = name
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Execute search
        
        Args:
            query: Search query
            limit: Maximum number of results
        
        Returns:
            List of search results
        """
        raise NotImplementedError


class WebSearchTool(SearchTool):
    """Web search tool powered by DuckDuckGo (ddgs)"""
    
    def __init__(self):
        super().__init__("WebSearch")
        from .web_search import web_search_tool
        self._tool = web_search_tool
        logger.info("Initialized WebSearchTool with DuckDuckGo backend")
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search the web
        
        Args:
            query: Search query
            limit: Maximum number of results
        
        Returns:
            List of search results with title, url, snippet
        """
        return self._tool.search(query, limit=limit)


class KnowledgeBaseTool(SearchTool):
    """Knowledge base search tool"""
    
    def __init__(self):
        super().__init__("KnowledgeBase")
        logger.info("Initialized KnowledgeBaseTool")
        
        # Initialize vector stores from retrieval module
        try:
            from src.retrieval import vector_store_manager
            self.vector_store_manager = vector_store_manager
        except Exception as e:
            logger.error(f"Error loading vector store manager: {e}")
            self.vector_store_manager = None
    
    def search(self, query: str, limit: int = 5, domain: str = None) -> List[Dict]:
        """
        Search knowledge base
        
        Args:
            query: Search query
            limit: Maximum number of results
            domain: Optional domain filter (sport, tech, general)
        
        Returns:
            List of search results
        """
        try:
            if not self.vector_store_manager:
                logger.error("Vector store manager not available")
                return []
            
            results = []
            
            # Search specified domain or all domains
            domains_to_search = [domain] if domain else ["sport", "tech", "general"]
            
            for search_domain in domains_to_search:
                docs = self.vector_store_manager.search(
                    domain=search_domain,
                    query=query,
                    k=limit
                )
                
                for i, doc in enumerate(docs):
                    results.append({
                        "title": doc.metadata.get("source", "Unknown"),
                        "content": doc.page_content,
                        "domain": search_domain,
                        "source": "knowledge_base",
                        "rank": i + 1
                    })
            
            return results[:limit]
        
        except Exception as e:
            logger.error(f"Error in knowledge base search: {e}")
            return []


class SportSearchTool(SearchTool):
    """Specialized sport search tool"""
    
    def __init__(self):
        super().__init__("SportSearch")
        self.kb_tool = KnowledgeBaseTool()
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search sport knowledge base"""
        return self.kb_tool.search(query, limit, domain="sport")


class TechSearchTool(SearchTool):
    """Specialized tech search tool"""
    
    def __init__(self):
        super().__init__("TechSearch")
        self.kb_tool = KnowledgeBaseTool()
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search tech knowledge base"""
        return self.kb_tool.search(query, limit, domain="tech")


class GeneralSearchTool(SearchTool):
    """Specialized general knowledge search tool"""
    
    def __init__(self):
        super().__init__("GeneralSearch")
        self.kb_tool = KnowledgeBaseTool()
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search general knowledge base"""
        return self.kb_tool.search(query, limit, domain="general")


# Tool registry
AVAILABLE_TOOLS = {
    "web_search": WebSearchTool,
    "knowledge_base": KnowledgeBaseTool,
    "sport_search": SportSearchTool,
    "tech_search": TechSearchTool,
    "general_search": GeneralSearchTool,
}


def get_tool(tool_name: str) -> Optional[SearchTool]:
    """Get a tool by name"""
    try:
        tool_class = AVAILABLE_TOOLS.get(tool_name)
        if tool_class:
            return tool_class()
        else:
            logger.warning(f"Unknown tool: {tool_name}")
            return None
    except Exception as e:
        logger.error(f"Error getting tool {tool_name}: {e}")
        return None
