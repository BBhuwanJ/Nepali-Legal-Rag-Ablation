#!/usr/bin/env python
"""
Step 4 — Compute Retrieval Metrics (Chunking Ablation)
=======================================================
Computes Hit@1, Hit@5, Precision@5, Recall@5, and MRR@5 for each
strategy's populated JSON.  No LLM calls needed — this is purely
algorithmic comparison of retrieved chunk dafa numbers vs required_dafas.

Requires: 02_populate_answers.py to have been run first.

Output:
  experiments/chunking_ablation/results/<strategy>_retrieval_metrics.json
  experiments/chunking_ablation/retrieval_metrics_summary.json

Usage:
  python 04_retrieval_metrics.py
  python 04_retrieval_metrics.py --strategy recursive
  python 04_retrieval_metrics.py --k 5
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="strict", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="strict", line_buffering=True)
    os.environ["PYTHONIOENCODING"] = "utf-8"

_THIS_DIR        = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent
_BACKEND_DIR     = _EXPERIMENTS_DIR.parent

for _p in [str(_BACKEND_DIR), str(_EXPERIMENTS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.retrieval_metrics import compute_dataset_metrics  # type: ignore[import]

_RESULTS_DIR = _THIS_DIR / "results"
STRATEGIES   = ["recursive", "dafa", "semantic", "hybrid"]


def compute_strategy(strategy: str, k: int) -> dict:
    populated_path = _RESULTS_DIR / f"{strategy}_populated.json"

    if not populated_path.exists():
        print(f"❌ Populated file not found: {populated_path}")
        print("   Run 02_populate_answers.py first.")
        return {"strategy": strategy, "error": "missing_populated_file"}

    print(f"\n{'='*65}")
    print(f"  Computing retrieval metrics for [{strategy}]  k={k}")
    print(f"{'='*65}")

    with open(populated_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    result = compute_dataset_metrics(eval_data, k=k)
    agg    = result['aggregate']

    # Save per-strategy results
    out_path = _RESULTS_DIR / f"{strategy}_retrieval_metrics.json"
    payload  = {
        "strategy":  strategy,
        "k":         k,
        "primary_split": result["primary_split"],
        "aggregation_policy": result["aggregation_policy"],
        "aggregate": agg,
        "aggregate_combined": result["aggregate_combined"],
        "by_split": result["by_split"],
        "refusal": result["refusal"],
        "per_question": result['per_question'],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"   Primary split: {result['primary_split']}")
    print(f"   Hit@1        : {agg.get('hit@1')}")
    print(f"   Hit@{k}       : {agg.get(f'hit@{k}')}")
    print(f"   Precision@{k} : {agg.get(f'precision@{k}')}")
    print(f"   Recall@{k}    : {agg.get(f'recall@{k}')}")
    print(f"   MRR@{k}       : {agg.get(f'mrr@{k}')}")
    print(f"   Questions    : {agg.get('n_evaluated')} evaluated, {agg.get('n_skipped')} skipped")
    print(f"   Combined N   : {result['aggregate_combined'].get('n_evaluated')}")
    print(f"   Refusal rate : {result['refusal'].get('refusal_compliance')}")
    print(f"💾 Saved: {out_path}")

    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Compute retrieval metrics for chunking ablation (no LLM needed)"
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES + ["all"],
        default="all",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Rank cutoff for metrics (default: 5)",
    )
    args = parser.parse_args()

    to_run = STRATEGIES if args.strategy == "all" else [args.strategy]
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n╔" + "═"*63 + "╗")
    print("║" + " "*7 + "LawBot IEEE — Retrieval Metrics (Chunking Ablation)" + " "*5 + "║")
    print("╚" + "═"*63 + "╝")
    print(f"\n  Strategies : {to_run}")
    print(f"  k          : {args.k}")

    all_results = {}
    split_results = {}
    for strategy in to_run:
        payload = compute_strategy(strategy, args.k)
        all_results[strategy] = payload.get("aggregate", {})
        split_results[strategy] = {
            "primary_split": payload.get("primary_split"),
            "by_split": payload.get("by_split", {}),
            "refusal": payload.get("refusal", {}),
        }

    # ── Summary table ────────────────────────────────────────────────────────
    k = args.k
    print("\n\n" + "="*75)
    print(f"📊  RETRIEVAL METRICS SUMMARY  (k={k})")
    print("="*75)
    hdr = (
        f"{'Strategy':<12}  {'Hit@1':>7}  {'Hit@'+str(k):>7}  "
        f"{'Prec@'+str(k):>7}  {'Rec@'+str(k):>7}  {'MRR@'+str(k):>7}  {'#Eval':>6}"
    )
    print(hdr)
    print("-"*75)
    for strategy in to_run:
        a = all_results.get(strategy, {})
        if "error" in a:
            print(f"{strategy:<12}  (error: {a['error']})")
            continue
        print(
            f"{strategy:<12}  "
            f"{str(a.get('hit@1','?')):>7}  "
            f"{str(a.get(f'hit@{k}','?')):>7}  "
            f"{str(a.get(f'precision@{k}','?')):>7}  "
            f"{str(a.get(f'recall@{k}','?')):>7}  "
            f"{str(a.get(f'mrr@{k}','?')):>7}  "
            f"{str(a.get('n_evaluated','?')):>6}"
        )
    print("="*75)

    # Save combined summary
    summary_path = _THIS_DIR / "retrieval_metrics_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    split_summary_path = _THIS_DIR / "retrieval_metrics_split_summary.json"
    with open(split_summary_path, 'w', encoding='utf-8') as f:
        json.dump(split_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Summary saved: {summary_path}")
    print("\n✅ Done! Next step: run 05_summarize_results.py")


if __name__ == "__main__":
    main()
