"""
Web Search & Web Scraper Tools - Powered by DuckDuckGo (ddgs) and BeautifulSoup
"""

import ipaddress
import logging
import socket
from typing import List, Dict, Optional
import httpx
from urllib.parse import urlparse
from bs4 import BeautifulSoup

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
    """Reject URLs that resolve to private/loopback/link-local addresses (SSRF guard)."""
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


class WebScraperTool:
    """Fetches and cleans main text content from a web page"""
    
    def __init__(self, timeout: int = 10, max_chars: int = 2500):
        self.timeout = timeout
        self.max_chars = max_chars
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
    
    def fetch_page_content(self, url: str) -> Optional[str]:
        """
        Download and extract clean text from a URL
        
        Args:
            url: The webpage URL
            
        Returns:
            Cleaned text snippet or None if failed
        """
        if not url or not url.startswith(("http://", "https://")):
            return None
        if not _is_public_url(url):
            logger.warning(f"Blocked non-public URL: {url}")
            return None
            
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True, headers=self.headers) as client:
                resp = client.get(url)
                if resp.status_code != 200:
                    logger.warning(f"Failed to fetch {url}: status {resp.status_code}")
                    return None
                    
                html = resp.text
                soup = BeautifulSoup(html, "html.parser")
                
                # Remove unwanted tags
                for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "svg"]):
                    tag.decompose()
                    
                # Extract text
                text = soup.get_text(separator=" ", strip=True)
                
                # Normalize whitespace
                cleaned = " ".join(text.split())
                if len(cleaned) > self.max_chars:
                    cleaned = cleaned[:self.max_chars] + "... [Trang bị cắt ngắn]"
                    
                return cleaned
                
        except Exception as e:
            logger.warning(f"Error scraping {url}: {e}")
            return None


# Global singleton instances
web_search_tool = WebSearchTool()
web_scraper_tool = WebScraperTool()
