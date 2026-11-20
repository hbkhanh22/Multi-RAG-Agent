"""
Retrieval metrics and result aggregation for the offline evaluation.

A dataset item lists the evidence that should be retrieved:
    "evidence": [{"source": "Laws of the Game", "pages": [54, 55]}]
`source` is a case-insensitive substring of the chunk's source/title/filename;
`pages` (0-based, as stored in chunk metadata) is optional.
"""

from statistics import mean
from typing import Any, Dict, List, Optional


def is_relevant(context: Dict[str, Any], evidence: List[Dict[str, Any]]) -> bool:
    haystack = " ".join(
        str(context.get(k) or "") for k in ("source", "title", "filename")
    ).lower()
    for spec in evidence:
        if str(spec.get("source", "")).lower() not in haystack:
            continue
        pages = spec.get("pages")
        if not pages or context.get("page") in pages:
            return True
    return False


def retrieval_metrics(contexts: List[Dict[str, Any]], evidence: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """hit / reciprocal rank / precision over the ranked contexts given to the LLM."""
    if not evidence:
        return None
    flags = [is_relevant(c, evidence) for c in contexts]
    first = next((i for i, f in enumerate(flags) if f), None)
    return {
        "k": len(contexts),
        "hit": first is not None,
        "reciprocal_rank": 0.0 if first is None else 1.0 / (first + 1),
        "precision": (sum(flags) / len(flags)) if flags else 0.0,
        "relevant_ranks": [i + 1 for i, f in enumerate(flags) if f],
    }


def _avg(values: List[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(mean(values), 3) if values else None


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-item results into one summary dict."""
    ok = [r for r in results if not r.get("error")]
    retr = [r["retrieval"] for r in ok if r.get("retrieval")]
    final = [r["final_verdict"] for r in ok if r.get("final_verdict")]
    initial = [r["online_initial"] for r in ok if r.get("online_initial")]
    online_final = [r["online_final"] for r in ok if r.get("online_final")]
    criteria = ("faithfulness", "relevance", "completeness", "correctness")

    def scores(verdicts, c):
        return _avg([v["scores"].get(c) for v in verdicts])

    refined = [r for r in ok if r.get("refine_rounds", 0) > 0]

    # RAGAS (0..1); a metric that errored / did not apply is None and is left out of the mean
    ragas_names: List[str] = []
    for r in ok:
        for name in (r.get("ragas") or {}):
            if name not in ragas_names:
                ragas_names.append(name)
    ragas = {}
    for name in ragas_names:
        vals = [(r.get("ragas") or {}).get(name) for r in ok]
        vals = [v for v in vals if v is not None]
        ragas[name] = {"mean": _avg(vals), "n": len(vals)}

    return {
        "ragas": ragas,
        "items": len(results),
        "errors": len(results) - len(ok),
        "route_accuracy": _avg([1.0 if r.get("route_ok") else 0.0 for r in ok if r.get("route_ok") is not None]),
        "retrieval": {
            "hit_rate": _avg([1.0 if m["hit"] else 0.0 for m in retr]),
            "mrr": _avg([m["reciprocal_rank"] for m in retr]),
            "precision": _avg([m["precision"] for m in retr]),
            "evaluated": len(retr),
        },
        "answer_final": {c: scores(final, c) for c in criteria} | {
            "overall": _avg([v["overall"] for v in final]),
            "pass_rate": _avg([1.0 if v["passed"] else 0.0 for v in final]),
        },
        "self_correction": {
            "judged_items": len(initial),
            "first_draft_overall": _avg([v["overall"] for v in initial]),
            "first_draft_pass_rate": _avg([1.0 if v["passed"] else 0.0 for v in initial]),
            "after_refine_overall": _avg([v["overall"] for v in online_final]),
            "after_refine_pass_rate": _avg([1.0 if v["passed"] else 0.0 for v in online_final]),
            "refined_items": len(refined),
            "fixed": sum(1 for r in refined if (r.get("online_final") or {}).get("passed")),
        },
        "latency_s": {
            "mean": _avg([r.get("latency_s") for r in ok]),
            "max": max((r.get("latency_s") or 0 for r in ok), default=None),
        },
    }
