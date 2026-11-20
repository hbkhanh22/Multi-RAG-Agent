"""
Quality gate for CI/CD: fail (exit 1) when retrieval quality drops below thresholds.

    python scripts/eval_gate.py evals/results/ci-sport evals/results/ci-tech --min-hit 0.9 --min-mrr 0.8

Each argument is a result directory (latest *-retrieval.json is used) or a json file.
Metrics are pooled over all questions of all given runs.
"""

import argparse
import glob
import json
import sys
from pathlib import Path


def latest(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    files = sorted(glob.glob(str(p / "*-retrieval.json")))
    if not files:
        raise SystemExit(f"no *-retrieval.json in {p}")
    return Path(files[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--min-hit", type=float, default=0.90)
    ap.add_argument("--min-mrr", type=float, default=0.80)
    ap.add_argument("--max-latency", type=float, default=0.0, help="mean seconds per question (0 = off)")
    a = ap.parse_args()

    rows = []
    for run in a.runs:
        rows += json.loads(latest(run).read_text(encoding="utf-8"))["results"]
    n = len(rows)
    hit = sum(1 for r in rows if r["retrieval"]["hit"]) / n
    mrr = sum(r["retrieval"]["reciprocal_rank"] for r in rows) / n
    lat = sum(r.get("latency_s", 0) for r in rows) / n
    print(f"questions {n}: hit {hit:.3f} (min {a.min_hit}) | MRR {mrr:.3f} (min {a.min_mrr}) | latency {lat:.2f}s")

    failed = []
    if hit < a.min_hit:
        failed.append("hit rate")
    if mrr < a.min_mrr:
        failed.append("MRR")
    if a.max_latency and lat > a.max_latency:
        failed.append("latency")
    if failed:
        print("GATE FAILED: " + ", ".join(failed))
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
