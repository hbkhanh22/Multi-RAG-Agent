"""Self-evaluation loop (judge -> refine) and offline evaluation metrics."""
import json

import pytest

from tests.conftest import MCP_QUERY
from src.evaluation.metrics import is_relevant, retrieval_metrics, summarize
from src.evaluation.judge import format_contexts, parse_json_object


# ------------------------------------------------------------------ metrics (pure)

CTX = [
    {"source_id": 1, "source": "data/raw/sport/FIFA Statutes 2024.pdf", "page": 6, "text": "a"},
    {"source_id": 2, "source": "data/raw/sport/Laws of the Game 2026_27.pdf", "page": 54, "text": "b"},
    {"source_id": 3, "source": "data/raw/sport/Laws of the Game 2026_27.pdf", "page": 10, "text": "c"},
]


def test_is_relevant_matches_source_and_page():
    assert is_relevant(CTX[1], [{"source": "laws of the game", "pages": [54]}])
    assert not is_relevant(CTX[2], [{"source": "Laws of the Game", "pages": [54]}])
    assert is_relevant(CTX[2], [{"source": "Laws of the Game"}])  # no pages => any page


def test_retrieval_metrics_rank_and_precision():
    m = retrieval_metrics(CTX, [{"source": "Laws of the Game", "pages": [54, 55]}])
    assert m["hit"] and m["reciprocal_rank"] == 0.5 and m["relevant_ranks"] == [2]
    assert abs(m["precision"] - 1 / 3) < 1e-9
    miss = retrieval_metrics(CTX, [{"source": "VFF"}])
    assert not miss["hit"] and miss["reciprocal_rank"] == 0.0
    assert retrieval_metrics(CTX, []) is None


def test_summarize_counts_refined_and_fixed():
    v = lambda o, p: {"scores": {"faithfulness": 4, "relevance": 4, "completeness": 4, "correctness": 4},
                      "overall": o, "passed": p}
    results = [
        {"id": "a", "route_ok": True, "retrieval": {"hit": True, "reciprocal_rank": 1.0, "precision": 0.5},
         "online_initial": v(2.0, False), "online_final": v(4.0, True), "refine_rounds": 1,
         "final_verdict": v(4.0, True), "latency_s": 2.0},
        {"id": "b", "route_ok": False, "retrieval": {"hit": False, "reciprocal_rank": 0.0, "precision": 0.0},
         "online_initial": v(5.0, True), "online_final": v(5.0, True), "refine_rounds": 0,
         "final_verdict": v(3.0, False), "latency_s": 4.0},
        {"id": "c", "error": "boom"},
    ]
    s = summarize(results)
    assert s["errors"] == 1 and s["route_accuracy"] == 0.5
    assert s["retrieval"]["hit_rate"] == 0.5 and s["retrieval"]["mrr"] == 0.5
    assert s["self_correction"]["refined_items"] == 1 and s["self_correction"]["fixed"] == 1
    assert s["self_correction"]["first_draft_pass_rate"] == 0.5
    assert s["self_correction"]["after_refine_pass_rate"] == 1.0
    assert s["answer_final"]["pass_rate"] == 0.5


def test_format_contexts_respects_budget_and_numbering():
    out = format_contexts([{"source_id": 7, "title": "T", "page": 3, "text": "x" * 50}], max_chars=1000)
    assert out.startswith("[Source 7] T (page 3)")
    assert len(format_contexts([{"source_id": 1, "text": "y" * 5000}], max_chars=100)) <= 100


def test_parse_json_object_variants():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object("[1, 2]") == {}
    assert parse_json_object("nope") == {}


# --------------------------------------------------------------- judge (fake LLM)

def test_judge_verdict_and_threshold(fake_llm):
    from src.graph.nodes import judge_agent
    fake_llm.judge_scores = [{"faithfulness": 2, "relevance": 5, "completeness": 4}]
    v = judge_agent.evaluate("q", "a", CTX)
    assert v["passed"] is False and v["failed_criteria"] == ["faithfulness"]
    assert v["overall"] == pytest.approx(11 / 3, abs=0.01) and v["issues"] == ["missing detail"]


def test_judge_unparseable_output_returns_none(fake_llm, monkeypatch):
    from src.graph.nodes import judge_agent
    monkeypatch.setattr(type(judge_agent), "invoke", lambda self, *a, **k: "not json")
    assert judge_agent.evaluate("q", "a", CTX) is None


# ------------------------------------------------------------ graph loop (end-to-end)

def run(query):
    from src.graph import run_workflow
    return run_workflow(query)


def test_good_answer_is_judged_once_and_kept(fake_llm, fake_web, tech_ready):
    r = run(MCP_QUERY)
    assert r["final_answer"] == "Docs say so [Source 1]"
    assert r["evaluation"]["passed"] is True and r["iteration"] == 0
    assert len(r["evaluation_history"]) == 1 and fake_llm.judge_calls == 1
    assert r["contexts"] and r["contexts"][0]["source_id"] == 1


def test_bad_answer_is_refined_with_feedback(fake_llm, fake_web, tech_ready):
    fake_llm.judge_scores = [
        {"faithfulness": 1, "relevance": 4, "completeness": 2},   # first draft fails
        {"faithfulness": 5, "relevance": 5, "completeness": 5},   # revision passes
    ]
    r = run(MCP_QUERY)
    assert r["final_answer"] == "REFINED [Source 1]"
    assert r["iteration"] == 1 and fake_llm.judge_calls == 2
    assert [h["passed"] for h in r["evaluation_history"]] == [False, True]
    assert r["sources"] and r["sources"][0]["source_id"] == 1 and "text" not in r["sources"][0]


def test_refine_rounds_are_capped(fake_llm, fake_web, tech_ready, monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "self_eval_max_rounds", 2)
    fake_llm.judge_scores = [{"faithfulness": 1, "relevance": 1, "completeness": 1}]  # never passes
    r = run(MCP_QUERY)
    assert r["iteration"] == 2 and fake_llm.judge_calls == 3
    assert r["status"] == "complete" and r["evaluation"]["passed"] is False


def test_self_eval_disabled_skips_judge(fake_llm, fake_web, tech_ready, monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "self_eval_enabled", False)
    r = run(MCP_QUERY)
    assert fake_llm.judge_calls == 0 and not r["evaluation"]


def test_stream_emits_reset_before_refined_tokens(fake_llm, fake_web, tech_ready):
    from src.graph import stream_workflow
    fake_llm.judge_scores = [{"faithfulness": 1, "relevance": 4, "completeness": 2},
                             {"faithfulness": 5, "relevance": 5, "completeness": 5}]
    events = list(stream_workflow(MCP_QUERY))
    types = [e["type"] for e in events]
    assert all(e.get("draft") for e in events if e["type"] == "token")  # nothing definitive before final
    reset = types.index("reset")
    assert "evaluation" in types[:reset]
    streamed_after = "".join(e["text"] for e in events[reset:] if e["type"] == "token")
    assert streamed_after == "REFINED [Source 1]"
    assert events[-1]["type"] == "final" and events[-1]["state"]["final_answer"] == streamed_after


def test_api_exposes_evaluation_without_contexts(fake_llm, fake_web, tech_ready):
    from fastapi.testclient import TestClient
    from api.main import app
    body = TestClient(app).post("/api/query", json={"query": MCP_QUERY}).json()  # no lifespan
    assert body["evaluation"]["passed"] is True and body["evaluation"]["refine_rounds"] == 0
    assert "contexts" not in json.dumps(body["agent_responses"])


# ------------------------------------------------------------------ RAGAS (no network)

def test_summarize_averages_ragas_ignoring_missing():
    results = [{"id": "a", "ragas": {"faithfulness": 1.0, "context_recall": None}},
               {"id": "b", "ragas": {"faithfulness": 0.5, "context_recall": 0.25}}]
    r = summarize(results)["ragas"]
    assert r["faithfulness"] == {"mean": 0.75, "n": 2}
    assert r["context_recall"] == {"mean": 0.25, "n": 1}


def test_ragas_scorer_skips_and_isolates_failures(monkeypatch):
    pytest.importorskip("ragas")
    from src.evaluation.ragas_eval import RagasScorer

    class Res:
        def __init__(self, v): self.value = v

    class Ok:
        async def ascore(self, **kw):
            assert "reference" not in kw or kw["reference"]
            return Res(0.8)

    class Boom:
        async def ascore(self, **kw):
            raise ValueError("bad json")

    scorer = RagasScorer(metrics=["faithfulness", "context_recall", "factual_correctness"])
    scorer.metrics = {"faithfulness": Ok(), "context_recall": Ok(), "factual_correctness": Boom()}
    out = scorer.score_many([
        {"id": "1", "question": "q", "answer": "a", "contexts": ["c"], "reference": "r"},
        {"id": "2", "question": "q", "answer": "a", "contexts": [], "reference": None},
    ])
    assert out[0]["scores"] == {"faithfulness": 0.8, "context_recall": 0.8, "factual_correctness": None}
    assert "bad json" in out[0]["errors"]["factual_correctness"]
    # no contexts / no reference => skipped, not reported as errors
    assert out[1]["scores"] == {"faithfulness": None, "context_recall": None, "factual_correctness": None}
    assert out[1]["errors"] == {}


def test_ragas_rejects_unknown_metric():
    pytest.importorskip("ragas")
    from src.evaluation.ragas_eval import RagasScorer
    with pytest.raises(ValueError):
        RagasScorer(metrics=["bleu"])


def test_report_renders_with_ragas_section():
    from src.evaluation.run import render_markdown
    results = [{"id": "a", "question": "q", "route": "sport", "route_ok": True, "refine_rounds": 0,
                "latency_s": 1.0, "retrieval": {"hit": True, "reciprocal_rank": 1.0, "precision": 1.0},
                "final_verdict": {"scores": {"faithfulness": 5, "relevance": 5, "completeness": 5},
                                  "overall": 5.0, "passed": True},
                "ragas": {"faithfulness": 0.9, "factual_correctness": 0.5}}]
    md = render_markdown(summarize(results), results, {"timestamp": "t", "datasets": ["d"], "self_eval": True,
                         "max_rounds": 1, "min_score": 4, "judge_model": "m", "ragas_model": "m"})
    assert "## RAGAS (0-1" in md and "| faithfulness | 0.900 | 1 |" in md and "## RAGAS per item" in md
