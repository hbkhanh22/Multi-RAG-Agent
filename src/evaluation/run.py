"""
Offline evaluation of the full pipeline on a labelled dataset.

    python -m src.evaluation.run --dataset evals/datasets/sport.jsonl
    python -m src.evaluation.run --dataset evals/datasets/sport.jsonl evals/datasets/tech.jsonl --limit 5
    python -m src.evaluation.run --dataset evals/datasets/tech.jsonl --no-refine   # baseline
    python -m src.evaluation.run --ragas                                            # + RAGAS metrics
    python -m src.evaluation.run --rescore evals/results/<run>.json                 # RAGAS on a past run

Dataset: JSON Lines, one item per line:
    {"id": "...", "question": "...", "domain": "sport|tech|general|web",
     "reference": "expected answer", "evidence": [{"source": "file substring", "pages": [0-based]}]}

Per item it records routing, retrieval metrics (hit / reciprocal rank / precision of the
chunks given to the LLM), the online judge verdicts before/after self-correction, and a
final judge verdict that also checks correctness against the reference answer.
With --ragas it adds RAGAS claim-level metrics (see ragas_eval.py).
Writes evals/results/<timestamp>.json and .md.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from config.settings import BASE_DIR, settings
from src.evaluation.metrics import retrieval_metrics, summarize

logger = logging.getLogger("evaluation")


def load_dataset(paths: List[Path]) -> List[Dict[str, Any]]:
    items = []
    for path in paths:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                item = json.loads(line)
                if not item.get("question"):
                    raise ValueError(f"{path}:{n}: missing 'question'")
                item.setdefault("id", f"{path.stem}-{n}")
                items.append(item)
    return items


def evaluate_item(item: Dict[str, Any], judge, run_workflow) -> Dict[str, Any]:
    question = item["question"]
    t0 = time.time()
    state = run_workflow(question)
    latency = round(time.time() - t0, 2)

    if state.get("status") != "complete":
        return {"id": item["id"], "question": question, "error": state.get("error") or "workflow failed",
                "latency_s": latency}

    contexts = state.get("contexts") or []
    agents = state.get("active_agents") or []
    domain = item.get("domain")
    history = state.get("evaluation_history") or []
    answer = state.get("final_answer", "")

    final_verdict = judge.evaluate(question, answer, contexts, reference=item.get("reference"))
    responses = state.get("agent_responses") or {}

    return {
        "id": item["id"],
        "question": question,
        "domain": domain,
        "route": state.get("route"),
        "agents": agents,
        "route_ok": (domain in agents) if domain else None,
        "fallback": next((r.get("fallback") for r in responses.values()
                          if isinstance(r, dict) and r.get("fallback")), None),
        "retrieval": retrieval_metrics(contexts, item.get("evidence") or []),
        "retrieved_sources": [
            {"rank": i, "source": Path(str(c.get("source") or "")).name, "page": c.get("page")}
            for i, c in enumerate(contexts, 1)
        ],
        "retrieved_contexts": [c.get("text") or "" for c in contexts],
        "online_initial": history[0] if history else None,
        "online_final": history[-1] if history else None,
        "refine_rounds": state.get("iteration", 0),
        "final_verdict": final_verdict,
        "answer": answer,
        "reference": item.get("reference"),
        "latency_s": latency,
    }


def evaluate_retrieval(item: Dict[str, Any], rag_agents: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieval only (expansion -> hybrid -> rerank -> fallbacks), no answer generation."""
    domain = item.get("domain") or "general"
    agent = rag_agents.get(domain) or rag_agents["general"]
    t0 = time.time()
    documents, _, fallback = agent._gather_context(item["question"])
    contexts = agent._contexts(documents)
    return {
        "id": item["id"],
        "question": item["question"],
        "domain": domain,
        "fallback": fallback,
        "retrieval": retrieval_metrics(contexts, item.get("evidence") or []),
        "retrieved_sources": [
            {"rank": i, "source": Path(str(c.get("source") or "")).name, "page": c.get("page")}
            for i, c in enumerate(contexts, 1)
        ],
        "latency_s": round(time.time() - t0, 2),
    }


def render_retrieval_markdown(results: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    s = summarize(results)["retrieval"]
    fb = [r.get("fallback") for r in results if not r.get("error")]
    lines = [
        f"# Retrieval report — {meta['timestamp']}",
        "",
        f"Index: `{meta['index']}` · chunk {meta['chunk_size']}/{meta['chunk_overlap']} · "
        f"contextual: {meta['contextual']} · cross-lingual expansion: {meta['crosslingual']} · "
        f"items: {len(results)}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Hit rate | {_fmt(s['hit_rate'])} |",
        f"| MRR | {_fmt(s['mrr'])} |",
        f"| Precision | {_fmt(s['precision'])} |",
        f"| Fallback used (not 'none') | {sum(1 for f in fb if f != 'none')} / {len(fb)} |",
        f"| Latency mean (s) | {_fmt(summarize(results)['latency_s']['mean'])} |",
        "",
        "| id | hit | RR | prec | fallback | retrieved (file p.page) |",
        "|---|---|---|---|---|---|",
    ]
    for x in results:
        if x.get("error"):
            lines.append(f"| {x['id']} | ERROR {x['error'][:60]} | | | | |")
            continue
        r = x.get("retrieval") or {}
        got = ", ".join(f"{d['source'][:18]} p{d['page']}" for d in x["retrieved_sources"])
        lines.append(f"| {x['id']} | {_fmt(r.get('hit'))} | {_fmt(r.get('reciprocal_rank'))} "
                     f"| {_fmt(r.get('precision'))} | {x['fallback']} | {got} |")
    return "\n".join(lines) + "\n"


def _fmt(v, nd=2):
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render_markdown(summary: Dict[str, Any], results: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    s = summary
    af, sc, r = s["answer_final"], s["self_correction"], s["retrieval"]
    lines = [
        f"# Evaluation report — {meta['timestamp']}",
        "",
        f"Datasets: {', '.join(meta['datasets'])} · items: {s['items']} · errors: {s['errors']} · "
        f"self-eval: {meta['self_eval']} (max rounds {meta['max_rounds']}, min score {meta['min_score']}) · "
        f"judge: {meta['judge_model']}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Route accuracy | {_fmt(s['route_accuracy'])} |",
        f"| Retrieval hit rate | {_fmt(r['hit_rate'])} |",
        f"| Retrieval MRR | {_fmt(r['mrr'])} |",
        f"| Retrieval precision | {_fmt(r['precision'])} |",
        f"| Faithfulness (1-5) | {_fmt(af['faithfulness'])} |",
        f"| Relevance (1-5) | {_fmt(af['relevance'])} |",
        f"| Completeness (1-5) | {_fmt(af['completeness'])} |",
        f"| Correctness vs reference (1-5) | {_fmt(af['correctness'])} |",
        f"| Final pass rate | {_fmt(af['pass_rate'])} |",
        f"| Online judge: first draft → after refine (overall) | "
        f"{_fmt(sc['first_draft_overall'])} → {_fmt(sc['after_refine_overall'])} |",
        f"| Online judge: first draft → after refine (pass rate) | "
        f"{_fmt(sc['first_draft_pass_rate'])} → {_fmt(sc['after_refine_pass_rate'])} |",
        f"| Answers refined / fixed | {sc['refined_items']} / {sc['fixed']} |",
        f"| Latency mean / max (s) | {_fmt(s['latency_s']['mean'])} / {_fmt(s['latency_s']['max'])} |",
    ]
    if s.get("ragas"):
        lines += [
            "",
            f"## RAGAS (0-1, LLM: {meta.get('ragas_model', '?')})",
            "",
            "| Metric | Mean | Scored items |",
            "|---|---|---|",
        ] + [f"| {m} | {_fmt(v['mean'], 3)} | {v['n']} |" for m, v in s["ragas"].items()]
    lines += [
        "",
        "## Items",
        "",
        "| id | route | hit | RR | faith | relev | compl | correct | refined | s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for x in results:
        if x.get("error"):
            lines.append(f"| {x['id']} | ERROR: {x['error'][:60]} | | | | | | | | {_fmt(x.get('latency_s'))} |")
            continue
        sc_ = (x.get("final_verdict") or {}).get("scores", {})
        rt = x.get("retrieval") or {}
        lines.append(
            f"| {x['id']} | {x['route']}{'' if x['route_ok'] in (True, None) else ' ✗'} "
            f"| {_fmt(rt.get('hit'))} | {_fmt(rt.get('reciprocal_rank'))} "
            f"| {_fmt(sc_.get('faithfulness'))} | {_fmt(sc_.get('relevance'))} "
            f"| {_fmt(sc_.get('completeness'))} | {_fmt(sc_.get('correctness'))} "
            f"| {x['refine_rounds']} | {_fmt(x['latency_s'])} |"
        )

    if s.get("ragas"):
        names = list(s["ragas"])
        lines += ["", "## RAGAS per item", "",
                  "| id | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for x in results:
            if x.get("ragas") is not None:
                lines.append(f"| {x['id']} | " + " | ".join(_fmt(x["ragas"].get(m), 3) for m in names) + " |")

    problems = [x for x in results if not x.get("error") and x.get("final_verdict")
                and not x["final_verdict"]["passed"]]
    if problems:
        lines += ["", "## Failing answers", ""]
        for x in problems:
            v = x["final_verdict"]
            lines += [f"### {x['id']}", "", f"**Q:** {x['question']}", "",
                      f"Failed: {', '.join(v['failed_criteria'])} · retrieved: "
                      + ", ".join(f"{s['source']} p{s['page']}" for s in x["retrieved_sources"]), ""]
            lines += [f"- {i}" for i in v.get("issues", [])] + [""]
    return "\n".join(lines) + "\n"


def add_ragas_scores(results: List[Dict[str, Any]], metrics: List[str] = None) -> str:
    """Score every successful result with RAGAS in place; return the RAGAS LLM model name."""
    from src.evaluation.ragas_eval import RagasScorer

    scorer = RagasScorer(metrics=metrics)
    todo = [r for r in results if not r.get("error")]
    if any(not r.get("retrieved_contexts") for r in todo):
        logger.warning("Some results have no retrieved_contexts: context-based RAGAS metrics are skipped for them")
    print(f"RAGAS: scoring {len(todo)} answers x {len(scorer.metric_names)} metrics "
          f"with {scorer.model} (concurrency {scorer.concurrency})...", flush=True)
    t0 = time.time()
    scored = scorer.score_many([
        {"id": r["id"], "question": r["question"], "answer": r.get("answer", ""),
         "contexts": r.get("retrieved_contexts") or [], "reference": r.get("reference")}
        for r in todo
    ])
    for r, sc in zip(todo, scored):
        r["ragas"] = sc["scores"]
        if sc["errors"]:
            r["ragas_errors"] = sc["errors"]
    print(f"RAGAS done in {time.time() - t0:.0f}s", flush=True)
    return scorer.model


def write_report(out_dir: Path, summary, results, meta, stem: str = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"{stem}.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = render_markdown(summary, results, meta)
    (out_dir / f"{stem}.md").write_text(report, encoding="utf-8")
    print("\n" + report.split("## Items")[0])
    print(f"Saved: {out_dir / (stem + '.md')}")
    return out_dir / f"{stem}.md"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the multi-RAG agent on a labelled dataset")
    parser.add_argument("--dataset", nargs="+", type=Path,
                        default=[BASE_DIR / "evals" / "datasets" / "sport.jsonl"])
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N items")
    parser.add_argument("--only", nargs="*", default=None, help="evaluate only these item ids")
    parser.add_argument("--no-refine", action="store_true", help="judge only, no self-correction (baseline)")
    parser.add_argument("--no-self-eval", action="store_true", help="disable the online judge entirely")
    parser.add_argument("--ragas", action="store_true", help="also compute RAGAS metrics")
    parser.add_argument("--ragas-metrics", nargs="+", default=None,
                        help="subset of: faithfulness answer_relevancy context_precision "
                             "context_recall factual_correctness")
    parser.add_argument("--rescore", type=Path, default=None,
                        help="add RAGAS metrics to an existing results .json (no pipeline run)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="only measure retrieval (fast: no answer generation, no judge)")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "evals" / "results")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.rescore:
        data = json.loads(args.rescore.read_text(encoding="utf-8"))
        results, meta = data["results"], data["meta"]
        meta["ragas_model"] = add_ragas_scores(results, args.ragas_metrics)
        meta["rescored_from"] = args.rescore.name
        write_report(args.out, summarize(results), results, meta)
        return 0

    if args.retrieval_only:
        items = load_dataset(args.dataset)
        if args.only:
            items = [i for i in items if i["id"] in set(args.only)]
        if args.limit:
            items = items[: args.limit]
        from src.agents.rag_agent import RAGAgent
        from src.retrieval.vector_store import embedding_slug
        agents = {d: RAGAgent(d) for d in ("sport", "tech", "general")}
        results = []
        for n, item in enumerate(items, 1):
            try:
                res = evaluate_retrieval(item, agents)
            except Exception as e:
                logger.exception(f"{item['id']} crashed")
                res = {"id": item["id"], "question": item["question"], "error": str(e)}
            results.append(res)
            print(f"[{n}/{len(items)}] {item['id']}: hit={(res.get('retrieval') or {}).get('hit')} "
                  f"rr={(res.get('retrieval') or {}).get('reciprocal_rank')} {res.get('latency_s')}s", flush=True)
        meta = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "retrieval-only",
            "datasets": [p.name for p in args.dataset],
            "index": embedding_slug(),
            "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap,
            "contextual": settings.contextual_chunks,
            "crosslingual": settings.query_expansion_crosslingual,
        }
        args.out.mkdir(parents=True, exist_ok=True)
        stem = datetime.now().strftime("%Y%m%d-%H%M%S") + "-retrieval"
        (args.out / f"{stem}.json").write_text(json.dumps(
            {"meta": meta, "summary": summarize(results), "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        report = render_retrieval_markdown(results, meta)
        (args.out / f"{stem}.md").write_text(report, encoding="utf-8")
        print("\n" + report)
        return 0

    if args.no_self_eval:
        settings.self_eval_enabled = False
    if args.no_refine:
        settings.self_eval_max_rounds = 0

    items = load_dataset(args.dataset)
    if args.only:
        items = [i for i in items if i["id"] in set(args.only)]
    if args.limit:
        items = items[: args.limit]

    # Heavy imports (models, vector stores) only once the arguments are valid
    from src.graph.workflow import run_workflow
    from src.graph.nodes import judge_agent

    results = []
    for n, item in enumerate(items, 1):
        print(f"[{n}/{len(items)}] {item['id']}: {item['question'][:70]}", flush=True)
        try:
            res = evaluate_item(item, judge_agent, run_workflow)
        except Exception as e:  # keep going: one bad item must not kill the run
            logger.exception(f"{item['id']} crashed")
            res = {"id": item["id"], "question": item["question"], "error": str(e)}
        results.append(res)
        v = res.get("final_verdict") or {}
        print(f"    route={res.get('route')} hit={(res.get('retrieval') or {}).get('hit')} "
              f"scores={v.get('scores')} refined={res.get('refine_rounds')} {res.get('latency_s')}s", flush=True)

    meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": [p.name for p in args.dataset],
        "self_eval": settings.self_eval_enabled,
        "max_rounds": settings.self_eval_max_rounds,
        "min_score": settings.self_eval_min_score,
        "judge_model": judge_agent.model,
        "answer_model": settings.default_llm_model,
    }
    # Save the pipeline results first: RAGAS is slow and rate-limited, and an interrupted
    # scoring run must not lose the answers (re-score later with --rescore <file>.json).
    stem = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary = summarize(results)
    write_report(args.out, summary, results, meta, stem=stem)

    if args.ragas or args.ragas_metrics:
        meta["ragas_model"] = add_ragas_scores(results, args.ragas_metrics)
        summary = summarize(results)
        write_report(args.out, summary, results, meta, stem=stem)  # same files, now with RAGAS
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
