"""
Web search: Tavily (official API, when TAVILY_API_KEY is set) with ddgs as fallback.

ddgs scrapes public search pages, which regularly answer with a captcha / 429 / 403, so
"0 results" from it cannot be told apart from "blocked": it is retried, and when nothing
answered the caller gets SearchUnavailable instead of an empty list. An empty list means
a search service really answered that it has no result.

Search snippets rarely hold the facts asked for (scores, prices), so the most relevant
passages of the top pages are added as `content`: from Tavily's raw content, or by
reading the page (see page_reader.py).
"""

import ipaddress
import logging
import re
import socket
import time
from typing import Dict, List, Optional
from urllib.parse import urlparse

import httpx

from config.settings import settings
from src.observability import metrics

logger = logging.getLogger(__name__)

_VIETNAMESE = re.compile(r"[ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ]", re.I)


class SearchUnavailable(Exception):
    """No search provider could answer (blocked, rate limited, quota, network)."""


class WebSearchTool:
    """Web search with retry, provider fallback and page reading."""

    def __init__(self, default_max_results: int = 5, timeout: int = 10, retry_delay: float = 1.0):
        self.default_max_results = default_max_results
        self.timeout = timeout
        self.retry_delay = retry_delay

    def search(self, query: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Search the live web.

        Returns results with title, url, snippet, content (relevant page passages, may
        be empty) and source_id. [] = the search worked but found nothing.
        Raises SearchUnavailable when no provider could answer.
        """
        query = (query or "").strip()
        if not query:
            return []
        limit = limit or self.default_max_results

        raw: List[Dict] = []
        answered = False
        for name, provider in self._providers():
            try:
                raw = provider(query, limit)
                answered = True
                metrics.record_web_search(name, "ok" if raw else "empty")
                logger.info(f"Web search ({name}) returned {len(raw)} results")
                if raw:
                    break
            except SearchUnavailable as e:
                metrics.record_web_search(name, "unavailable")
                logger.warning(f"Web search provider {name} unavailable: {e}")
        if not answered:
            raise SearchUnavailable("no web search provider answered")
        if not raw:
            return []

        results = [
            {
                "title": r["title"] or f"Web Result #{i}",
                "url": r["url"],
                "snippet": r["snippet"],
                "content": r.get("content", ""),
                "source": r["url"],
                "source_id": i,
                "domain": "web",
                "type": "web_search",
                "rank": i,
            }
            for i, r in enumerate(raw, 1)
        ]
        self._add_page_content(query, results)
        return results

    # ------------------------------------------------------------- providers

    def _providers(self):
        if settings.tavily_api_key:
            yield "tavily", self._tavily
        yield "ddgs", self._ddgs

    def _tavily(self, query: str, limit: int) -> List[Dict]:
        body = {
            "query": query[:400],
            "max_results": limit,
            "search_depth": settings.tavily_search_depth,
            "include_raw_content": "text",
        }
        if _VIETNAMESE.search(query):
            body["country"] = "vietnam"      # boosts, does not filter

        error = ""
        for attempt in range(2):
            if attempt:
                time.sleep(self.retry_delay)
            try:
                resp = httpx.post(
                    f"{settings.tavily_base_url}/search", json=body, timeout=settings.tavily_timeout,
                    headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
                )
                if resp.status_code == 200:
                    return [self._tavily_result(query, r) for r in resp.json().get("results") or [] if r.get("url")]
            except (httpx.HTTPError, ValueError, AttributeError) as e:
                error = type(e).__name__
                continue
            error = f"HTTP {resp.status_code}"
            if resp.status_code != 429 and resp.status_code < 500:
                break       # bad key (401), plan limit (432/433)...: retrying cannot help
        raise SearchUnavailable(error)

    @staticmethod
    def _tavily_result(query: str, r: Dict) -> Dict:
        from src.tools.page_reader import excerpt

        return {
            "title": (r.get("title") or "").strip(),
            "url": r["url"],
            "snippet": (r.get("content") or "").strip(),
            "content": excerpt(query, r.get("raw_content") or "", settings.web_page_chars),
        }

    def _ddgs(self, query: str, limit: int) -> List[Dict]:
        from ddgs import DDGS

        error = ""
        for attempt in range(max(1, settings.web_search_attempts)):
            if attempt:
                time.sleep(self.retry_delay)
            try:
                items = list(DDGS(timeout=self.timeout).text(query, max_results=limit))
            except Exception as e:      # includes "No results found" (= blocked or really none)
                error = f"{type(e).__name__}: {e}"[:200]
                continue
            found = [
                {"title": (i.get("title") or "").strip(), "url": (i.get("href") or "").strip(),
                 "snippet": (i.get("body") or "").strip()}
                for i in items if i.get("href")
            ]
            if found:
                return found
            error = "empty response"
        raise SearchUnavailable(error)

    # ------------------------------------------------------------ page content

    @staticmethod
    def _add_page_content(query: str, results: List[Dict]) -> None:
        """Read the top pages that came without content (best effort, time-boxed)."""
        from src.tools.page_reader import read_pages

        todo = [r for r in results[: settings.web_read_pages] if not r["content"]]
        if not todo:
            return
        try:
            texts = read_pages([r["url"] for r in todo], query)
        except Exception as e:
            logger.warning(f"Page reading failed: {e}")
            return
        for r in todo:
            r["content"] = texts.get(r["url"], "")


def result_text(result: Dict) -> str:
    """What the LLM and the judge see of one search result."""
    snippet, content = result.get("snippet") or "", result.get("content") or ""
    return f"{snippet}\n{content}".strip() if content else snippet


def _is_public_url(url: str) -> bool:
    """Reject URLs that resolve to private/loopback/link-local addresses (SSRF guard).
    Used before reading a page and by the MCP manager before a tool fetches a URL."""
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
