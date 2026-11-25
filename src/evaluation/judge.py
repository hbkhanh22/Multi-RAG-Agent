"""
LLM-as-judge and answer refinement.

AnswerJudge scores an answer against the context it was generated from:
  - faithfulness : every claim is supported by the context (no hallucination)
  - relevance    : the answer addresses the question that was asked
  - completeness : the answer covers what the context offers for the question
  - correctness  : (offline only) agreement with a reference answer
Scores are 1-5. The judge also returns concrete feedback, which AnswerRefiner
feeds back to the LLM to produce a revised answer (reflection loop).
"""

from typing import Any, Dict, List, Optional
import json
import logging
import re

from src.agents.base_agent import BaseAgent
from config.settings import settings

logger = logging.getLogger(__name__)

ONLINE_CRITERIA = ("faithfulness", "relevance", "completeness")


def parse_json_object(raw: str) -> Dict[str, Any]:
    """Parse a JSON object from LLM output, tolerating ```json fences / prose."""
    text = (raw or "").strip()
    try:
        data = json.loads(text, strict=False)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0), strict=False)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def format_contexts(contexts: List[Dict[str, Any]], max_chars: Optional[int] = None) -> str:
    """Render contexts as the same [Source N] blocks the answering LLM saw."""
    budget = max_chars or settings.judge_max_context_chars
    if not max_chars and any(c.get("agent") == "files" for c in contexts):
        # the judge must see what the File Agent saw, or true statements look unsupported
        budget = max(budget, settings.attachment_context_chars * 3 // 2)
    parts: List[str] = []
    used = 0
    for c in contexts:
        title = c.get("title") or c.get("source") or "Unknown"
        page = f" (page {c['page']})" if c.get("page") is not None else ""
        block = f"[Source {c.get('source_id')}] {title}{page}\n{c.get('text', '')}\n"
        if used + len(block) > budget:
            block = block[: max(0, budget - used)]
        if not block:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


def _score(value: Any) -> Optional[int]:
    try:
        return max(1, min(5, int(round(float(value)))))
    except (TypeError, ValueError):
        return None


class AnswerJudge(BaseAgent):
    """Grades an answer (1-5 per criterion) and explains what to fix."""

    def __init__(self):
        super().__init__(
            name="Answer Judge",
            model=settings.judge_llm_model or settings.default_llm_model,
            temperature=0.0,
            max_tokens=settings.judge_max_tokens,
        )

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.evaluate(
            input_data.get("query", ""),
            input_data.get("answer", ""),
            input_data.get("contexts") or [],
            input_data.get("reference"),
        ) or {}

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[Dict[str, Any]],
        reference: Optional[str] = None,
        min_score: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Return a verdict dict, or None if the judge could not produce one
        (callers must treat None as "unknown", never as a failure).
        """
        threshold = settings.self_eval_min_score if min_score is None else min_score
        criteria = list(ONLINE_CRITERIA) + (["correctness"] if reference else [])

        system_prompt = """You are a strict evaluator of answers produced by a retrieval-augmented (RAG) assistant.
Grade the ANSWER to the QUESTION using ONLY the CONTEXT (and the REFERENCE answer, if given).

Criteria, each scored 1-5 (5 = excellent, 3 = acceptable with flaws, 1 = unacceptable):
- faithfulness: every factual claim is supported by the CONTEXT. Claims the context does not
  support lower this score, unless the answer explicitly says they come from general knowledge.
  Wrong [Source N] attributions also lower it.
- relevance: the answer directly addresses what was asked, without drifting off-topic.
- completeness: the answer covers the key information the CONTEXT offers for this question.
- correctness (only when a REFERENCE is given): the answer agrees with the reference answer.

Return ONLY valid JSON, no markdown fences:
{
  "scores": {"faithfulness": 1-5, "relevance": 1-5, "completeness": 1-5, "correctness": 1-5 or null},
  "issues": ["short description of each concrete problem"],
  "feedback": "actionable instructions to fix the answer (empty string if nothing to fix)",
  "context_sufficient": true or false
}
"context_sufficient" is false when the CONTEXT simply lacks the information needed to answer."""

        prompt = (
            f"QUESTION:\n{query}\n\n"
            f"CONTEXT:\n{format_contexts(contexts) if contexts else '(no context retrieved)'}\n\n"
            + (f"REFERENCE ANSWER:\n{reference}\n\n" if reference else "")
            + f"ANSWER:\n{answer}\n\nRespond with JSON only:"
        )

        try:
            data = parse_json_object(self.invoke([prompt], system_prompt=system_prompt))
        except Exception as e:
            logger.warning(f"Judge call failed, skipping evaluation: {e}")
            return None

        raw_scores = data.get("scores") if isinstance(data.get("scores"), dict) else {}
        scores = {c: _score(raw_scores.get(c)) for c in criteria}
        if any(scores[c] is None for c in ONLINE_CRITERIA):
            logger.warning(f"Judge returned unparseable scores: {str(data)[:200]}")
            return None

        issues = data.get("issues") if isinstance(data.get("issues"), list) else []
        failed = [c for c in criteria if scores.get(c) is not None and scores[c] < threshold]
        valid = [s for s in scores.values() if s is not None]
        return {
            "scores": scores,
            "overall": round(sum(valid) / len(valid), 2),
            "passed": not failed,
            "failed_criteria": failed,
            "issues": [str(i) for i in issues][:8],
            "feedback": str(data.get("feedback") or "").strip(),
            "context_sufficient": bool(data.get("context_sufficient", True)),
            "judge_model": self.model,
        }


class AnswerRefiner(BaseAgent):
    """Rewrites an answer using the judge's feedback (one reflection step)."""

    def __init__(self):
        super().__init__(name="Answer Refiner", model=settings.default_llm_model, temperature=0.0)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"answer": self.refine(**input_data)}

    def refine(
        self,
        query: str,
        answer: str,
        contexts: List[Dict[str, Any]],
        verdict: Dict[str, Any],
        on_token=None,
    ) -> str:
        system_prompt = """You are revising an answer written by a RAG assistant after a reviewer found problems.
Write an improved, complete answer to the QUESTION that fixes every issue in the REVIEW.
Rules:
- Answer in the same language as the QUESTION.
- Use only facts from the CONTEXT. If the context does not contain something the question needs,
  say so explicitly instead of guessing (you may add general knowledge only if labelled as such).
- Cite with the existing [Source N] markers exactly as numbered in the CONTEXT; never invent numbers.
- Output only the revised answer: no preamble, no mention of the review."""

        issues = "\n".join(f"- {i}" for i in verdict.get("issues", [])) or "- (none listed)"
        prompt = (
            f"QUESTION:\n{query}\n\n"
            f"CONTEXT:\n{format_contexts(contexts) if contexts else '(no context retrieved)'}\n\n"
            f"PREVIOUS ANSWER:\n{answer}\n\n"
            f"REVIEW (scores 1-5): {json.dumps(verdict.get('scores', {}))}\n"
            f"Issues:\n{issues}\n"
            f"How to fix: {verdict.get('feedback') or 'address the issues above'}\n\n"
            "Revised answer:"
        )
        return self.invoke([prompt], system_prompt=system_prompt, on_token=on_token).strip()
