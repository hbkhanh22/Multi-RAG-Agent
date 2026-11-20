"""
Web search (DuckDuckGo via ddgs) and the SSRF guard shared with the MCP fetch tool.
Fetching page content is done by the MCP "fetch" server (see src/mcp_client).
"""

import ipaddress
import logging
import socket
from typing import List, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class WebSearchTool:
    """DuckDuckGo web search tool using ddgs"""
    
    def __init__(self, default_max_results: int = 5, timeout: int = 10):
        self.default_max_results = default_max_results
        self.timeout = timeout
        logger.info("Initialized WebSearchTool with DuckDuckGo backend")
    
    def search(self, query: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Search the live web using DuckDuckGo
        
        Args:
            query: Search query string
            limit: Maximum number of search results (defaults to default_max_results)
            
        Returns:
            List of search results with title, url, snippet, and domain info
        """
        max_results = limit or self.default_max_results
        results: List[Dict] = []
        
        if not query or not query.strip():
            return results
            
        try:
            # Use ddgs package
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
                
            ddgs_client = DDGS(timeout=self.timeout)
            raw_results = list(ddgs_client.text(query.strip(), max_results=max_results))
            
            for idx, item in enumerate(raw_results):
                title = item.get("title", "").strip()
                href = item.get("href", "").strip()
                snippet = item.get("body", "").strip()
                
                results.append({
                    "title": title or f"Web Result #{idx + 1}",
                    "url": href,
                    "snippet": snippet,
                    "source": href,
                    "source_id": idx + 1,
                    "domain": "web",
                    "type": "web_search",
                    "rank": idx + 1
                })
                
            logger.info(f"Web search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error executing web search for '{query}': {e}")
            return []


def _is_public_url(url: str) -> bool:
    """Reject URLs that resolve to private/loopback/link-local addresses (SSRF guard).
    Used by the MCP manager before letting a tool fetch a URL."""
    try:
        host = urlparse(url).hostname
        if not host:
            return False
        for info in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(info[4][0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
                return False
        return True
    except Exception:
        return False


# Global singleton instance
web_search_tool = WebSearchTool()
