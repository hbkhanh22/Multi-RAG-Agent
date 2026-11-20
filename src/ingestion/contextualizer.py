"""
Contextual chunk headers (a page-level take on Anthropic's "Contextual Retrieval").

A 512-1000 character chunk such as "• at least 9.15 m (10 yds) from the penalty mark"
says nothing about which document, law or article it belongs to, so neither the
embedding nor BM25 can match it to "khoảng cách khi đá phạt đền". Before chunking,
each page gets ONE LLM call that returns:
  - title   : the document title (asked on the first page only, reused afterwards)
  - section : the chapter / law / article the page belongs to (the previous page's
              section is passed in, so pages without a heading inherit it)
  - summary : one sentence describing what the page covers
Every chunk of the page is then prefixed with "title › section\nsummary".

One call per page instead of per chunk keeps the cost low (~370 calls for the
current corpus vs ~1500); results are cached on disk keyed by file + size + mtime
+ page, so re-chunking or rebuilding the index never re-calls the LLM.
"""

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from config.settings import settings

logger = logging.getLogger(__name__)

CACHE_FILE = Path(settings.data_dir) / "vectorstores" / ".page_contexts.json"

_SYSTEM_PROMPT = """You label pages of a document so that search can find them.
Return ONLY valid JSON, no markdown fences:
{"title": "<document title>", "section": "<chapter / law / article / heading this page belongs to>",
 "summary": "<one sentence, max 30 words, saying what this page covers>"}
Rules:
- Write title, section and summary in the SAME language as the page text.
- "section": the most specific heading the page falls under, including its number
  (e.g. "Law 11 - Offside", "Article 9 - Official languages", "Luật 12 - Lỗi và hành vi khiếm nhã").
  If the page has no heading of its own, keep the PREVIOUS SECTION.
- "summary": concrete topics and terms on the page, not generic phrases like "this page discusses"."""


def _clean_title(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"[_]+", " ", stem).strip()


class PageContextualizer:
    def __init__(self):
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        self.model = settings.contextual_llm_model or settings.fast_llm_model
        self.llm = ChatNVIDIA(
            model=self.model,
            api_key=settings.nvidia_api_key,
            temperature=0.0,
            max_tokens=300,
            timeout=settings.llm_timeout,
            **({} if settings.llm_enable_thinking
               else {"model_kwargs": {"chat_template_kwargs": {"enable_thinking": False}}}),
        )
        self._lock = threading.Lock()
        self._cache = self._load_cache()

    # ---------------------------------------------------------------- cache

    @staticmethod
    def _load_cache() -> Dict[str, Any]:
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.warning(f"Page-context cache unreadable, starting empty: {e}")
            return {}

    def _save_cache(self) -> None:
        with self._lock:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = CACHE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(CACHE_FILE)

    # ----------------------------------------------------------------- core

    def _ask(self, page_text: str, prev_section: str, title: Optional[str]) -> Dict[str, str]:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = (
            (f"DOCUMENT TITLE: {title}\n" if title else "DOCUMENT TITLE: (unknown - infer it from the page)\n")
            + f"PREVIOUS SECTION: {prev_section or '(none)'}\n\n"
            + f"PAGE TEXT:\n{page_text[: settings.contextual_max_page_chars]}\n\nJSON:"
        )
        raw = self.llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]).content
        from src.evaluation.judge import parse_json_object  # shared tolerant JSON parser
        data = parse_json_object(raw)
        return {k: str(data.get(k) or "").strip()[:200] for k in ("title", "section", "summary")}

    def contextualize(self, pages: List[Document], file_key: str, size: int, mtime: float) -> None:
        """Annotate page documents in place with metadata doc_title / section / page_summary."""
        title: Optional[str] = None
        prev_section = ""
        filename = pages[0].metadata.get("filename", "") if pages else ""
        dirty = False

        for i, page in enumerate(pages):
            key = f"{file_key}|{size}|{int(mtime)}|{page.metadata.get('page', i)}"
            ctx = self._cache.get(key)
            if ctx is None:
                text = page.page_content.strip()
                if not text:
                    ctx = {"title": title or "", "section": prev_section, "summary": ""}
                else:
                    try:
                        ctx = self._ask(text, prev_section, title)
                    except Exception as e:
                        logger.warning(f"Context LLM failed for {filename} page {i}: {e}")
                        ctx = {"title": title or "", "section": prev_section, "summary": ""}
                with self._lock:
                    self._cache[key] = ctx
                dirty = True
                if dirty and i % 10 == 9:
                    self._save_cache()

            title = title or ctx.get("title") or _clean_title(filename)
            prev_section = ctx.get("section") or prev_section
            page.metadata["doc_title"] = title
            page.metadata["section"] = prev_section
            page.metadata["page_summary"] = ctx.get("summary", "")

        if dirty:
            self._save_cache()
        logger.info(f"Contextualised {len(pages)} pages of {filename} (title: {title!r})")


def chunk_header(meta: Dict[str, Any]) -> str:
    """'Title › Section\\nSummary' (empty parts dropped)."""
    head = " › ".join(p for p in (meta.get("doc_title"), meta.get("section")) if p)
    summary = meta.get("page_summary") or ""
    return "\n".join(p for p in (head, summary) if p)


def add_chunk_headers(chunks: List[Document]) -> List[Document]:
    """Prefix each chunk with its page's contextual header (embedding + BM25 + LLM all see it)."""
    for c in chunks:
        header = chunk_header(c.metadata)
        if header:
            c.metadata["context_header"] = header
            c.page_content = f"{header}\n\n{c.page_content}"
    return chunks
