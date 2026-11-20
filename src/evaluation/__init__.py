"""
Evaluation: LLM-as-judge, answer refinement (self-correction loop) and
offline retrieval/answer metrics. Run the offline suite with:

    python -m src.evaluation.run --dataset evals/datasets/sport.jsonl
"""

from .judge import AnswerJudge, AnswerRefiner, format_contexts, parse_json_object
from .metrics import is_relevant, retrieval_metrics, summarize

__all__ = [
    "AnswerJudge",
    "AnswerRefiner",
    "format_contexts",
    "parse_json_object",
    "is_relevant",
    "retrieval_metrics",
    "summarize",
]
