"""
File Agent - answers from the files the user attached to the conversation.

Everything that fits the context budget is given to the model whole. Longer files:
  * overview questions (summarise...)  each part is condensed in parallel, then combined
  * specific questions                 BM25 picks the passages (the query variants include a
                                       vi<->en translation)
Images are also looked at again with the actual question in mind.
"""

import contextvars
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from .base_agent import BaseAgent
from config.settings import settings
from src.attachments import attachment_store, vision
from src.guardrails.injection import DATA_RULE, neutralise
from src.memory.history import _norm
from src.observability import metrics
from src.runtime import Cancelled

logger = logging.getLogger(__name__)

CHUNK_CHARS = 1500
CHUNK_OVERLAP = 150
MAX_PARTS = 8            # parts a long file is condensed in, for an overview question

_OVERVIEW = re.compile(
    r"\b(tom tat|tom luoc|tong quan|khai quat|y chinh|noi dung chinh|diem chinh|noi (?:ve )?(?:cai )?gi|"
    r"noi dung (?:cua )?(?:file|tep|tai lieu|van ban)|phan tich (?:file|tep|tai lieu|van ban)|"
    r"doc (?:giup |cho )?(?:toi |minh )?(?:file|tep|tai lieu)|"
    r"summar\w*|overview|main (?:points?|ideas?)|key (?:points?|takeaways?)|what is (?:this|the) (?:file|document))\b"
)

Context = Dict[str, Any]


def _in_parallel(fn, items: List, workers: int = 4) -> List:
    """Ordered map in threads that keep the caller's context (usage, cancellation, tracing)."""
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        futures = [pool.submit(contextvars.copy_context().run, fn, item) for item in items]
        return [f.result() for f in futures]


def _tokens(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class PartReader(BaseAgent):
    """Condenses one part of a long file (small fast model)."""

    def __init__(self):
        super().__init__(name="File Part Reader", model=settings.fast_llm_model, temperature=0.0, max_tokens=900)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"notes": self.notes(input_data["query"], input_data["text"])}

    def notes(self, query: str, text: str) -> str:
        return self.invoke(
            [f"User request: {query}\n\nPart of the document:\n{text}\n\nNotes:"],
            system_prompt="Condense this part of a document into detailed bullet notes in the document's "
                          "language. Keep every name, number, date and conclusion that matters for the user "
                          "request. Only use the text; no introduction.\n" + DATA_RULE,
        ).strip()


class FileAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="File Agent", model=settings.default_llm_model, temperature=0.0)
        self.part_reader = PartReader()
        self.domain = "files"

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Needs 'query' and 'attachments' (metadata dicts already checked for ownership)."""
        query = input_data.get("query", "")
        base = {"domain": self.domain, "sources": [], "retrieved_count": 0, "reranked_count": 0}
        try:
            files = attachment_store.load([a["id"] for a in input_data.get("attachments") or []])
            if not files:
                return {**base, "success": False, "answer": "Không tìm thấy tệp đính kèm nào trong hội thoại này."}
            candidates = self._chunks(files)
            contexts = self._select(query, input_data.get("expanded_queries") or [], candidates)
            contexts = self._sanitise(self._with_observations(query, files, contexts))
            answer = self._generate(query, files, contexts, input_data.get("history", ""), input_data.get("on_token"))
            return {
                **base,
                "success": True,
                "answer": answer,
                "sources": [{**{k: v for k, v in c.items() if k != "text"}, "snippet": c["text"][:200]}
                            for c in contexts if f"[Source {c['source_id']}]" in answer],
                "contexts": contexts,
                "retrieved_count": len(candidates),
                "reranked_count": len(contexts),
                "model": self.model,
            }
        except Cancelled:
            raise
        except Exception as e:
            logger.error(f"Error in FileAgent: {e}", exc_info=True)
            return {**base, "success": False, "answer": "Đã xảy ra lỗi khi đọc tệp đính kèm.", "error": str(e)}

    # ------------------------------------------------------------- contexts

    @staticmethod
    def _context(f: Dict[str, Any], text: str, page=None, label=None) -> Context:
        return {"source": f"attachment:{f['id']}" + (f"#{label}" if label else ""),
                "title": f["filename"] + (f" · {label}" if label else ""), "page": page, "text": text}

    def _chunks(self, files: List[Dict[str, Any]]) -> List[Context]:
        out: List[Context] = []
        for f in files:
            for seg in f["segments"]:
                text, start = seg["text"], 0
                while start < len(text):
                    end = start + CHUNK_CHARS
                    if end < len(text):           # cut at a line / word end when there is one nearby
                        cut = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
                        end = cut if cut > start + CHUNK_CHARS // 2 else end
                    out.append(self._context(f, text[start:end], seg.get("page"), seg.get("label")))
                    if end >= len(text):
                        break
                    start = end - CHUNK_OVERLAP
        return out

    def _select(self, query: str, variants: List[str], chunks: List[Context]) -> List[Context]:
        budget = settings.attachment_context_chars
        if sum(len(c["text"]) for c in chunks) <= budget:
            return chunks
        if _OVERVIEW.search(_norm(query)):
            return self._condense(query, chunks)

        bm25 = BM25Okapi([_tokens(c["title"] + " " + c["text"]) or [""] for c in chunks])
        scores = [0.0] * len(chunks)
        for q in dict.fromkeys([query, *variants]):
            s = bm25.get_scores(_tokens(q))
            top = max(s)
            if top > 0:
                scores = [max(a, float(b) / top) for a, b in zip(scores, s)]
        picked, used = {0}, len(chunks[0]["text"])      # the beginning says what the file is
        for i in sorted(range(len(chunks)), key=lambda i: -scores[i]):
            if i not in picked and used + len(chunks[i]["text"]) <= budget:
                picked.add(i)
                used += len(chunks[i]["text"])
        logger.info(f"File Agent: {len(picked)}/{len(chunks)} passages selected")
        return [chunks[i] for i in sorted(picked)]      # reading order

    def _condense(self, query: str, chunks: List[Context]) -> List[Context]:
        """One context of notes per part, covering every file from start to end."""
        size = max(sum(len(c["text"]) for c in chunks) // MAX_PARTS + 1, 8000)
        parts: List[List[Context]] = []
        for c in chunks:
            last = parts[-1] if parts else None
            if (last is None or last[0]["source"].split("#")[0] != c["source"].split("#")[0]
                    or sum(len(x["text"]) for x in last) + len(c["text"]) > size):
                parts.append([c])
            else:
                last.append(c)
        share = settings.attachment_context_chars // len(parts)

        def condense(part: List[Context]) -> Context:
            text, _ = neutralise("\n".join(c["text"] for c in part))
            try:
                notes = self.part_reader.notes(query, text)
            except Cancelled:
                raise
            except Exception as e:
                logger.warning(f"Condensing a part failed, using its beginning: {e}")
                notes = ""
            first, last = part[0], part[-1]
            title = first["title"]
            if first["page"] is not None and last["page"] != first["page"]:
                title += f" (trang {first['page'] + 1}–{last['page'] + 1})"
            return {**first, "title": title, "text": (notes or text)[:share]}

        return _in_parallel(condense, parts)

    def _with_observations(self, query: str, files: List[Dict[str, Any]],
                           contexts: List[Context]) -> List[Context]:
        """Look at every image again for this question; the observation joins the image's
        own source (one image = one source to cite)."""
        def look(f: Dict[str, Any]) -> str:
            try:
                return vision.observe(f["image"], query)
            except vision.VisionUnavailable:
                return ""                         # the transcription made at upload remains

        images = [f for f in files if f.get("image")]
        contexts = list(contexts)
        for f, seen in zip(images, _in_parallel(look, images)):
            if not seen:
                continue
            i = next((i for i, c in enumerate(contexts) if c["source"] == f"attachment:{f['id']}"), None)
            if i is None:                         # its text was not among the selected passages
                contexts.append(self._context(f, seen))
            else:
                contexts[i] = {**contexts[i], "text": f"{seen}\n\n{contexts[i]['text']}"}
        return contexts

    @staticmethod
    def _sanitise(contexts: List[Context]) -> List[Context]:
        out, removed = [], 0
        for i, c in enumerate(contexts, 1):
            text, n = neutralise(c["text"])
            removed += n
            out.append({**c, "source_id": i, "text": text})
        if removed:
            logger.warning(f"Neutralised {removed} injection-like phrase(s) in attachments")
            metrics.record_injection("files", removed)
        return out

    # ------------------------------------------------------------ generation

    def _generate(self, query, files, contexts, history, on_token) -> str:
        system_prompt = (
            "You are the File Agent: you answer questions about the files the user attached.\n"
            "- Answer in the language of the question, in clear Markdown.\n"
            "- Use only the file contents below. If they do not contain the answer, say so plainly; add "
            "general knowledge only if clearly labelled as such.\n"
            "- End every sentence or bullet that states something from a file with the marker of its source, "
            "[Source N]. Do not add a list of sources and do not repeat the answer.\n"
            "- Copy numbers, names and dates exactly. Only excerpts of long files may be shown: never claim "
            "something is absent from the whole file, say it was not found in the excerpts read.\n"
            + DATA_RULE
        )
        blocks = "\n".join(
            f"[Source {c['source_id']}] {c['title']}"
            + (f" (page {c['page'] + 1})" if c["page"] is not None else "") + f"\n{c['text']}\n"
            for c in contexts
        )
        prompt = (
            (f"Earlier conversation (for context only; do not cite it):\n{history}\n\n" if history else "")
            + f"Attached files: {', '.join(f['filename'] for f in files)}\n\nFile contents:\n{blocks}\n"
            f"Question: {query}\n\nAnswer, with [Source N] markers inside the text:"
        )
        return self.invoke([prompt], system_prompt=system_prompt, on_token=on_token)
