"""
Synthesizer Agent - Produces the final answer from one or more agent responses.

- Single agent  -> its answer and citations are returned untouched (no extra LLM
  call, no truncation, [Source N] markers stay valid).
- Multiple agents -> source numbers are made globally unique, then ONE LLM call
  merges the answers and extracts key themes.
"""

from typing import Any, Dict, List, Tuple
import json
import logging
import re

from .base_agent import BaseAgent
from config.settings import settings

logger = logging.getLogger(__name__)

_SOURCE_RE = re.compile(r"\[Source (\d+)\]")
MAX_CHARS_PER_AGENT = 6000


class SynthesizerAgent(BaseAgent):
    """Synthesizes final answers from multiple sources"""

    def __init__(self):
        super().__init__(
            name="Answer Synthesizer",
            model=settings.default_llm_model,
            temperature=0.0,
        )

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            input_data: Must contain 'query' and 'agent_responses'
        """
        try:
            query = input_data.get("query", "")
            agent_responses = input_data.get("agent_responses", {})

            if not agent_responses:
                return {
                    "success": False,
                    "synthesized_answer": "Không có phản hồi nào từ các agent.",
                    "sources": [],
                    "retrieval_stats": {},
                }

            stats = self._collect_stats(agent_responses)
            usable = {
                name: r for name, r in agent_responses.items()
                if isinstance(r, dict) and r.get("success") and r.get("answer")
            }

            if not usable:
                return {
                    "success": False,
                    "query": query,
                    "synthesized_answer": self._failure_message(agent_responses),
                    "key_themes": [],
                    "sources": [],
                    "retrieval_stats": stats,
                }

            if len(usable) == 1:
                name, resp = next(iter(usable.items()))
                answer = resp["answer"]
                themes: List[str] = []
                citations = [{**c, "agent": name} for c in resp.get("sources", []) if isinstance(c, dict)]
            else:
                answers, citations = self._merge_with_global_numbering(usable)
                answer, themes = self._synthesize_combined(query, answers)

            return {
                "success": True,
                "query": query,
                "synthesized_answer": answer,
                "key_themes": themes,
                "sources": citations,
                "source_count": len(citations),
                "agent_count": len(usable),
                "retrieval_stats": stats,
            }

        except Exception as e:
            logger.error(f"Error in synthesizer: {e}", exc_info=True)
            return {
                "success": False,
                "synthesized_answer": "Đã xảy ra lỗi khi tổng hợp câu trả lời.",
                "sources": [],
                "retrieval_stats": {},
                "error": str(e),
            }

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _collect_stats(agent_responses: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        """Retrieval statistics per agent (kept separate from citations)."""
        return {
            name: {
                "retrieved": int(r.get("retrieved_count", 0) or 0),
                "reranked": int(r.get("reranked_count", 0) or 0),
            }
            for name, r in agent_responses.items() if isinstance(r, dict)
        }

    @staticmethod
    def _failure_message(agent_responses: Dict[str, Any]) -> str:
        details = [
            r.get("answer") for r in agent_responses.values()
            if isinstance(r, dict) and r.get("answer")
        ]
        base = "Không thể tạo câu trả lời cho câu hỏi này."
        return base + ("\n\n" + "\n".join(details) if details else "")

    @staticmethod
    def _merge_with_global_numbering(
        usable: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        """
        Each agent numbers its sources from 1. Shift every agent's numbers by a
        running offset so [Source N] is unique across the merged answer.
        """
        offset = 0
        answers: Dict[str, str] = {}
        citations: List[Dict[str, Any]] = []

        for name, resp in usable.items():
            sources = [c for c in resp.get("sources", []) if isinstance(c, dict)]
            width = max(
                int(resp.get("reranked_count") or 0),
                max((int(c.get("source_id") or 0) for c in sources), default=0),
            )

            answers[name] = _SOURCE_RE.sub(
                lambda m, off=offset: f"[Source {int(m.group(1)) + off}]", resp["answer"]
            )
            for c in sources:
                sid = c.get("source_id")
                citations.append({
                    **c,
                    "agent": name,
                    "source_id": int(sid) + offset if sid is not None else None,
                })
            offset += width

        return answers, citations

    def _synthesize_combined(self, query: str, answers: Dict[str, str]) -> Tuple[str, List[str]]:
        """One LLM call -> (answer, themes)."""
        formatted = ""
        for name, content in answers.items():
            formatted += f"\n[AGENT: {name.upper()}]\n{content[:MAX_CHARS_PER_AGENT]}\n" + "-" * 60 + "\n"

        system_prompt = """You are an expert at merging answers from several specialist agents.

Return ONLY valid JSON (no markdown fences, no extra text):
{
  "answer": "<merged answer, markdown allowed>",
  "themes": ["theme 1", "theme 2", "theme 3"]
}

Rules for "answer":
- Answer in the same language as the user's question.
- Integrate all agents' insights, remove redundancy, flag contradictions.
- Keep the citation markers like [Source 3] EXACTLY as they appear in the agent
  answers. Never invent, renumber or drop them.
- Use clear headings and structure.
"themes" must contain 3-5 short key points."""

        prompt = f"Query: {query}\n\nAgent answers:\n{formatted}\nRespond with JSON only:"

        try:
            raw = self.invoke([prompt], system_prompt=system_prompt)
            data = self._parse_json(raw)

            answer = str(data.get("answer", "")).strip()
            themes = data.get("themes", [])
            themes = [str(t) for t in themes][:5] if isinstance(themes, list) else []

            if not answer:
                return self._simple_synthesis(answers), themes
            return answer, themes

        except Exception as e:
            logger.error(f"Error in combined synthesis: {e}")
            return self._simple_synthesis(answers), []

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        """Parse a JSON object, tolerating markdown fences / surrounding text."""
        text = (raw or "").strip()
        try:
            return json.loads(text, strict=False)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return {}
            try:
                return json.loads(m.group(0), strict=False)
            except json.JSONDecodeError:
                return {}

    @staticmethod
    def _simple_synthesis(answers: Dict[str, str]) -> str:
        """Fallback: concatenate agent answers."""
        parts = ["## Tổng hợp từ nhiều nguồn\n"]
        for name, content in answers.items():
            parts.append(f"### {name.capitalize()}\n{content}\n")
        return "\n".join(parts)
