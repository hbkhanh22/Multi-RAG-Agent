"""
Read web pages found by the search and keep the passages relevant to the query.

Plain HTTP + HTML parsing: pages rendered by JavaScript yield little or no text.

SSRF: URLs come from search results, so every hop (redirects are followed by hand) must
be http(s) to a public address, and the address actually connected to is checked again
before the body is read (a host name may resolve differently the second time).
"""

import ipaddress
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from lxml import html as lxml_html

from config.settings import settings

logger = logging.getLogger(__name__)

MAX_BYTES = 2_000_000
MAX_REDIRECTS = 4
BLOCK_CHARS = 350
_UA = "Mozilla/5.0 (compatible; MultiRAGAgent/1.0)"
_DROP = "//script|//style|//noscript|//nav|//footer|//aside|//form|//svg|//iframe|//template|//comment()"
_BREAKS = {"p", "div", "li", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "table", "ul", "dt", "dd"}
_WORD = re.compile(r"\w+", re.UNICODE)


def _peer_is_public(resp: httpx.Response) -> bool:
    stream = resp.extensions.get("network_stream")
    addr = stream.get_extra_info("server_addr") if stream is not None else None
    if not addr:
        return False
    return ipaddress.ip_address(addr[0].split("%")[0]).is_global


def decode(body: bytes, content_type: str = "") -> str:
    """Page text: charset of the HTTP header, else of the <meta> tag, else UTF-8.
    (libxml2 falls back to Latin-1 when the <meta> tag comes late, garbling Vietnamese.)"""
    m = re.search(r"charset=[\"']?([\w-]+)", content_type, re.I)
    declared = m.group(1) if m else ""
    if not declared:
        m = re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", body, re.I)
        declared = m.group(1).decode("ascii", "ignore") if m else "utf-8"
    try:
        return body.decode(declared, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _fetch(url: str, deadline: float) -> Optional[str]:
    from src.tools.web_search import _is_public_url

    # trust_env=False: through a proxy the peer check would look at the proxy
    with httpx.Client(timeout=settings.web_page_timeout, trust_env=False, follow_redirects=False,
                      headers={"User-Agent": _UA, "Accept": "text/html"}) as client:
        for _ in range(MAX_REDIRECTS + 1):
            if urlparse(url).scheme not in ("http", "https") or not _is_public_url(url):
                logger.warning("Page skipped: not a public http(s) address")
                return None
            with client.stream("GET", url) as resp:
                if not _peer_is_public(resp):
                    logger.warning("Page skipped: connected to a non-public address")
                    return None
                if resp.is_redirect:
                    url = urljoin(url, resp.headers.get("location", ""))
                    continue
                if resp.status_code != 200 or "html" not in resp.headers.get("content-type", ""):
                    return None
                body = b""
                for part in resp.iter_bytes():
                    body += part
                    if len(body) > MAX_BYTES or time.monotonic() > deadline:
                        break
                return decode(body, resp.headers.get("content-type", ""))
    return None


def html_to_text(page: str) -> str:
    """Visible text: one line per block, table cells separated by " | "."""
    if not page.strip():
        return ""
    try:
        # bytes + explicit encoding: lxml rejects str with an XML encoding declaration
        root = lxml_html.document_fromstring(page.replace("\x00", "").encode("utf-8", "replace"),
                                             parser=lxml_html.HTMLParser(encoding="utf-8"))
    except Exception:
        return ""
    for el in root.xpath(_DROP):
        if el.getparent() is not None:
            el.drop_tree()
    parts: List[str] = []

    def walk(el):
        tag = el.tag if isinstance(el.tag, str) else ""
        if tag in _BREAKS:
            parts.append("\n")
        if el.text:
            parts.append(el.text)
        for child in el:
            walk(child)
            if child.tail:
                parts.append(child.tail)
        if tag in ("td", "th"):
            parts.append(" | ")
        elif tag in _BREAKS:
            parts.append("\n")
        else:
            parts.append(" ")

    try:
        walk(root)
    except RecursionError:
        return root.text_content()
    lines = (re.sub(r"[ \t\r\f\v\n\xa0]+", " ", ln).strip(" |") for ln in "".join(parts).split("\n"))
    return "\n".join(ln for ln in lines if ln)


def _recent_dates(days: int = 3) -> "re.Pattern[str]":
    """Matches today and the last `days` days: 22/11, 22/11/2026, 22-11-2026, 2026-11-22."""
    now = datetime.now(timezone(timedelta(hours=settings.local_utc_offset)))
    alts = []
    for d in (now - timedelta(days=i) for i in range(days + 1)):
        alts.append(rf"0?{d.day}[/.-]0?{d.month}(?:[/.-](?:{d.year}|{d.year % 100:02d}))?(?![\d/.-]*\d)")
        alts.append(rf"{d.year}-{d.month:02d}-{d.day:02d}")
    return re.compile(r"(?<![\d/.-])(?:" + "|".join(alts) + ")")


def excerpt(query: str, text: str, max_chars: int) -> str:
    """The blocks of `text` that best match the query (recent dates first), in reading order."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    blocks: List[str] = []
    cur = ""
    for line in text.split("\n"):
        for piece in (line[i:i + BLOCK_CHARS] for i in range(0, len(line), BLOCK_CHARS)):
            if cur and len(cur) + len(piece) > BLOCK_CHARS:
                blocks.append(cur)
                cur = ""
            cur = f"{cur}\n{piece}" if cur else piece
    if cur:
        blocks.append(cur)

    terms = {w for w in _WORD.findall(query.lower()) if len(w) > 1}

    recent = _recent_dates()

    def score(block: str) -> float:
        words = _WORD.findall(block.lower())
        # figures (scores, prices, dates) are usually what a web question is after
        return len(terms & set(words)) + min(sum(w.isdigit() for w in words), 6) * 0.25             + (3 if recent.search(block) else 0)

    order = sorted(range(len(blocks)), key=lambda i: (-score(blocks[i]), i))
    chosen, used = [], 0
    for i in order:
        if used + len(blocks[i]) + 1 > max_chars:
            continue
        chosen.append(i)
        used += len(blocks[i]) + 1
    return "\n".join(blocks[i] for i in sorted(chosen))


def _read(url: str, query: str, deadline: float) -> str:
    try:
        body = _fetch(url, deadline)
        return excerpt(query, html_to_text(body), settings.web_page_chars) if body else ""
    except Exception as e:
        logger.info(f"Could not read a page: {type(e).__name__}")
        return ""


def read_pages(urls: List[str], query: str) -> Dict[str, str]:
    """{url: relevant text} for the pages read within the time budget."""
    if not urls:
        return {}
    budget = settings.web_page_timeout + 2
    deadline = time.monotonic() + budget
    pool = ThreadPoolExecutor(max_workers=len(urls))
    futures = {pool.submit(_read, u, query, deadline): u for u in urls}
    done, _ = wait(futures, timeout=budget)
    pool.shutdown(wait=False, cancel_futures=True)   # a slow page must not hold the answer
    return {futures[f]: f.result() for f in done}
