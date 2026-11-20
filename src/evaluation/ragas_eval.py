"""
RAGAS metrics (https://docs.ragas.io) for the offline evaluation.

Unlike the 1-5 judge in judge.py, RAGAS decomposes answers into atomic claims
and verifies each one, producing 0..1 scores that are comparable with
published RAG benchmarks:

  faithfulness        share of the answer's claims supported by the retrieved contexts
  answer_relevancy    similarity between the question and questions regenerated from the
                      answer (LLM + embeddings); low = evasive / off-topic answer
  context_precision   are the chunks useful for the reference answer ranked first? (needs reference)
  context_recall      share of the reference answer's claims found in the contexts (needs reference)
  factual_correctness claim-level F1 between answer and reference answer (needs reference)

The LLM is NVIDIA NIM through its OpenAI-compatible API; the embeddings reuse the
bge-m3 model already loaded by the vector store (no second 2 GB copy).
ragas is an optional dependency: it is imported only when RAGAS scoring is requested.
"""

import asyncio
import logging
import math
import os
from typing import Any, Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# ragas sends anonymous usage telemetry by default: opt out unless the user set it
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

RAGAS_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "factual_correctness",
)
NEEDS_REFERENCE = {"context_precision", "context_recall", "factual_correctness"}
NEEDS_CONTEXTS = {"faithfulness", "context_precision", "context_recall"}
RATE_LIMIT_ATTEMPTS = 4
RATE_LIMIT_BASE_DELAY = 20  # seconds; doubles each attempt


def _is_rate_limit(err: Exception) -> bool:
    text = f"{type(err).__name__} {err}"
    return "429" in text or "Too Many Requests" in text or "RateLimit" in text


def _make_embeddings():
    """Adapt the app's LangChain bge-m3 embeddings to ragas' embedding interface."""
    from ragas.embeddings.base import BaseRagasEmbedding
    from src.retrieval import vector_store_manager

    lc = vector_store_manager.embeddings

    class _AppEmbeddings(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs: Any) -> List[float]:
            return lc.embed_query(text)

        async def aembed_text(self, text: str, **kwargs: Any) -> List[float]:
            return await asyncio.to_thread(lc.embed_query, text)

        def embed_texts(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            return lc.embed_documents(texts)

        async def aembed_texts(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            return await asyncio.to_thread(lc.embed_documents, texts)

    return _AppEmbeddings()


def _make_llm():
    import instructor
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    # max_retries: the OpenAI client retries 429/5xx with exponential backoff (honours Retry-After)
    client = AsyncOpenAI(base_url=settings.nvidia_base_url, api_key=settings.nvidia_api_key,
                         timeout=settings.llm_timeout, max_retries=6)
    extra = {} if settings.llm_enable_thinking else {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
    }
    model = settings.ragas_llm_model or settings.judge_llm_model or settings.default_llm_model
    # MD_JSON: parse JSON from the reply text, no response_format support needed
    return llm_factory(model, client=client, mode=instructor.Mode.MD_JSON,
                       temperature=0.0, max_tokens=settings.ragas_max_tokens, **extra), model


class RagasScorer:
    """Scores samples {question, answer, contexts: [str], reference?} with RAGAS metrics."""

    def __init__(self, metrics: Optional[List[str]] = None, concurrency: Optional[int] = None):
        try:
            from ragas.metrics.collections import (
                AnswerRelevancy, ContextPrecisionWithReference, ContextRecall,
                FactualCorrectness, Faithfulness,
            )
        except ImportError as e:  # pragma: no cover
            raise ImportError("RAGAS is not installed: pip install 'ragas>=0.4'") from e

        unknown = set(metrics or []) - set(RAGAS_METRICS)
        if unknown:
            raise ValueError(f"Unknown RAGAS metrics: {sorted(unknown)} (choose from {RAGAS_METRICS})")
        self.metric_names = list(metrics or RAGAS_METRICS)
        self.concurrency = concurrency or settings.ragas_concurrency

        llm, self.model = _make_llm()
        builders = {
            "faithfulness": lambda: Faithfulness(llm=llm),
            "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=_make_embeddings()),
            "context_precision": lambda: ContextPrecisionWithReference(llm=llm),
            "context_recall": lambda: ContextRecall(llm=llm),
            "factual_correctness": lambda: FactualCorrectness(llm=llm, mode="f1"),
        }
        self.metrics = {name: builders[name]() for name in self.metric_names}

    @staticmethod
    def _kwargs(name: str, s: Dict[str, Any]) -> Dict[str, Any]:
        q, a, ctx, ref = s["question"], s["answer"], s.get("contexts") or [], s.get("reference")
        return {
            "faithfulness": {"user_input": q, "response": a, "retrieved_contexts": ctx},
            "answer_relevancy": {"user_input": q, "response": a},
            "context_precision": {"user_input": q, "reference": ref, "retrieved_contexts": ctx},
            "context_recall": {"user_input": q, "retrieved_contexts": ctx, "reference": ref},
            "factual_correctness": {"response": a, "reference": ref},
        }[name]

    async def _score_metric(self, sem: asyncio.Semaphore, name: str, sample: Dict[str, Any]):
        if name in NEEDS_REFERENCE and not sample.get("reference"):
            return name, None, "no reference"
        if name in NEEDS_CONTEXTS and not sample.get("contexts"):
            return name, None, "no contexts"
        for attempt in range(RATE_LIMIT_ATTEMPTS):
            async with sem:
                try:
                    result = await self.metrics[name].ascore(**self._kwargs(name, sample))
                    value = float(result.value)
                    if math.isnan(value):
                        return name, None, "nan"
                    return name, round(value, 4), None
                except Exception as e:
                    err = e
            # Rate limited even after the client's own retries: wait outside the semaphore
            if _is_rate_limit(err) and attempt + 1 < RATE_LIMIT_ATTEMPTS:
                delay = RATE_LIMIT_BASE_DELAY * 2 ** attempt
                logger.info(f"RAGAS {name} rate-limited for {sample.get('id')}, retrying in {delay}s")
                await asyncio.sleep(delay)
                continue
            break
        logger.warning(f"RAGAS {name} failed for {sample.get('id')}: {type(err).__name__}: {err}")
        return name, None, f"{type(err).__name__}: {str(err)[:200]}"

    async def ascore_many(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(self.concurrency)
        jobs = [
            [self._score_metric(sem, name, s) for name in self.metric_names]
            for s in samples
        ]
        per_sample = await asyncio.gather(*(asyncio.gather(*j) for j in jobs))
        out = []
        for triples in per_sample:
            scores = {name: value for name, value, _ in triples}
            errors = {name: err for name, _, err in triples
                      if err and err not in ("no reference", "no contexts")}
            out.append({"scores": scores, "errors": errors})
        return out

    def score_many(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return asyncio.run(self.ascore_many(samples))
