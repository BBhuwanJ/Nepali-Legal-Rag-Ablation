#!/usr/bin/env python
"""
Step 4 — Compute Retrieval Metrics (Retrieval Ablation)
========================================================
Computes Hit@1, Hit@5, Precision@5, Recall@5, and MRR@5 for each
retrieval mode's populated JSON.  No LLM calls required.

Requires: 02_populate_answers.py to have been run first.

Output:
  experiments/retrieval_ablation/results/<mode>_retrieval_metrics.json
  experiments/retrieval_ablation/retrieval_metrics_summary.json

Usage:
  python 04_retrieval_metrics.py
  python 04_retrieval_metrics.py --retriever faiss_only
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
RETRIEVERS   = ["faiss_only", "bm25_only", "hybrid", "routed_hybrid"]

RETRIEVER_LABELS = {
    "faiss_only": "Semantic FAISS Only",
    "bm25_only":  "BM25 Only",
    "hybrid":     "Hybrid (ours)",
    "routed_hybrid": "Source-Aware Routed Hybrid (proposed)",
}


def compute_retriever(mode: str, k: int) -> dict:
    populated_path = _RESULTS_DIR / f"{mode}_populated.json"

    if not populated_path.exists():
        print(f"❌ Populated file not found: {populated_path}")
        print("   Run 02_populate_answers.py first.")
        return {"retriever": mode, "error": "missing_populated_file"}

    print(f"\n{'='*65}")
    print(f"  Computing retrieval metrics for [{mode}]  k={k}")
    print(f"{'='*65}")

    with open(populated_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    result = compute_dataset_metrics(eval_data, k=k)
    agg    = result['aggregate']

    out_path = _RESULTS_DIR / f"{mode}_retrieval_metrics.json"
    payload  = {
        "retriever":     mode,
        "label":         RETRIEVER_LABELS.get(mode, mode),
        "k":             k,
        "primary_split": result["primary_split"],
        "aggregation_policy": result["aggregation_policy"],
        "aggregate":     agg,
        "aggregate_combined": result["aggregate_combined"],
        "by_split": result["by_split"],
        "refusal": result["refusal"],
        "per_question":  result['per_question'],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"   Primary split: {result['primary_split']}")
    print(f"   Hit@1        : {agg.get('hit@1')}")
    print(f"   Hit@{k}       : {agg.get(f'hit@{k}')}")
    print(f"   Precision@{k} : {agg.get(f'precision@{k}')}")
    print(f"   Recall@{k}    : {agg.get(f'recall@{k}')}")
    print(f"   MRR@{k}       : {agg.get(f'mrr@{k}')}")
    print(f"   Complete@{k}  : {agg.get(f'complete@{k}')}")
    print(f"   Questions    : {agg.get('n_evaluated')} evaluated, {agg.get('n_skipped')} skipped")
    print(f"   Combined N   : {result['aggregate_combined'].get('n_evaluated')}")
    print(f"   Refusal rate : {result['refusal'].get('refusal_compliance')}")
    print(f"💾 Saved: {out_path}")

    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Compute retrieval metrics for retrieval ablation (no LLM needed)"
    )
    parser.add_argument(
        "--retriever",
        choices=RETRIEVERS + ["all"],
        default="all",
    )
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    to_run = RETRIEVERS if args.retriever == "all" else [args.retriever]
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n╔" + "═"*63 + "╗")
    print("║" + " "*6 + "LawBot IEEE — Retrieval Metrics (Retrieval Ablation)" + " "*5 + "║")
    print("╚" + "═"*63 + "╝")
    print(f"\n  Retrievers : {to_run}")
    print(f"  k          : {args.k}")
    print(f"  Index      : hybrid legal-first (constant)")

    all_results = {}
    split_results = {}
    for mode in to_run:
        payload = compute_retriever(mode, args.k)
        all_results[mode] = payload.get("aggregate", {})
        split_results[mode] = {
            "primary_split": payload.get("primary_split"),
            "by_split": payload.get("by_split", {}),
            "refusal": payload.get("refusal", {}),
        }

    # ── Summary table ────────────────────────────────────────────────────────
    k = args.k
    print("\n\n" + "="*80)
    print(f"📊  RETRIEVAL METRICS SUMMARY — RETRIEVAL ABLATION  (k={k})")
    print("="*80)
    hdr = (
        f"{'Retriever':<22}  {'Hit@1':>7}  {'Hit@'+str(k):>7}  "
        f"{'Prec@'+str(k):>8}  {'Rec@'+str(k):>7}  {'MRR@'+str(k):>7}  {'#Eval':>6}"
    )
    print(hdr)
    print("-"*80)
    for mode in to_run:
        a   = all_results.get(mode, {})
        lbl = RETRIEVER_LABELS.get(mode, mode)
        if "error" in a:
            print(f"{lbl:<22}  (error: {a['error']})")
            continue
        print(
            f"{lbl:<22}  "
            f"{str(a.get('hit@1','?')):>7}  "
            f"{str(a.get(f'hit@{k}','?')):>7}  "
            f"{str(a.get(f'precision@{k}','?')):>8}  "
            f"{str(a.get(f'recall@{k}','?')):>7}  "
            f"{str(a.get(f'mrr@{k}','?')):>7}  "
            f"{str(a.get('n_evaluated','?')):>6}"
        )
    print("="*80)

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
