#!/usr/bin/env python
"""
Step 5 — Summarize & Format Results (Retrieval Ablation)
=========================================================
Reads the retrieval metric files and produces IEEE-formatted tables
for inclusion in the paper.

Tables generated:
  TABLE VI  — Retrieval Strategy Comparison (Hybrid Legal-First Chunks, k=5)
  TABLE VII — Answer Quality (Correctness, Groundedness, etc.)

Output:
  experiments/retrieval_ablation/ieee_tables.txt
  experiments/retrieval_ablation/ieee_tables.json

Usage:
  python 05_summarize_results.py
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="strict", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="strict", line_buffering=True)
    os.environ["PYTHONIOENCODING"] = "utf-8"

_THIS_DIR    = Path(__file__).resolve().parent
_RESULTS_DIR = _THIS_DIR / "results"

RETRIEVERS = ["faiss_only", "bm25_only", "hybrid", "routed_hybrid"]

RETRIEVER_LABELS = {
    "faiss_only": "Semantic FAISS Only",
    "bm25_only":  "BM25 Only",
    "hybrid":     "Hybrid Dafa+BM25+Semantic",
    "routed_hybrid": "Source-Aware Routed Hybrid (proposed)",
}


def _load_json(path: Path):
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def _safe(d, *keys, default="—"):
    for k in keys:
        if d is None:
            return default
        d = d.get(k)
        if d is None:
            return default
    if isinstance(d, float):
        return f"{d:.4f}"
    return str(d)


def build_tables():
    # ── TABLE VI — Retrieval Strategy Comparison ────────────────────────────────
    ret_summary = _load_json(_THIS_DIR / "retrieval_metrics_summary.json") or {}
    k = 5

    tbl6 = []
    tbl6.append(f"\n{'='*80}")
    tbl6.append("TABLE VI — Held-Out Test Retrieval Strategy Comparison")
    tbl6.append(f"         (Hybrid Legal-First Chunks — constant, k={k})")
    tbl6.append(f"{'='*80}")
    hdr = (
        f"{'Retrieval Strategy':<26}  {'Hit@1':>7}  {'Hit@5':>7}  "
        f"{'Prec@5':>8}  {'Rec@5':>7}  {'MRR@5':>7}  {'N':>5}"
    )
    tbl6.append(hdr)
    tbl6.append("-"*80)

    for mode in RETRIEVERS:
        a   = ret_summary.get(mode, {})
        lbl = RETRIEVER_LABELS.get(mode, mode)
        if not a:
            m_json = _load_json(_RESULTS_DIR / f"{mode}_retrieval_metrics.json")
            if m_json:
                a = m_json.get("aggregate", {})
        if not a:
            tbl6.append(f"{lbl:<26}  (data not found — run 04_retrieval_metrics.py)")
            continue
        tbl6.append(
            f"{lbl:<26}  "
            f"{_safe(a, 'hit@1'):>7}  "
            f"{_safe(a, f'hit@{k}'):>7}  "
            f"{_safe(a, f'precision@{k}'):>8}  "
            f"{_safe(a, f'recall@{k}'):>7}  "
            f"{_safe(a, f'mrr@{k}'):>7}  "
            f"{str(a.get('n_evaluated','?')):>5}"
        )

    tbl6.append(f"{'='*80}")
    tbl6.append(f"Bolded row = best performer per column (apply manually in LaTeX)")

    tbl6b = []
    tbl6b.append(f"\n{'='*80}")
    tbl6b.append("TABLE VI-B — Combined 350-Question Retrieval Metrics (Supplementary)")
    tbl6b.append(f"{'='*80}")
    tbl6b.append(hdr)
    tbl6b.append("-"*80)
    for mode in RETRIEVERS:
        lbl = RETRIEVER_LABELS.get(mode, mode)
        payload = _load_json(_RESULTS_DIR / f"{mode}_retrieval_metrics.json") or {}
        a = payload.get("aggregate_combined", {})
        tbl6b.append(
            f"{lbl:<26}  "
            f"{_safe(a, 'hit@1'):>7}  "
            f"{_safe(a, f'hit@{k}'):>7}  "
            f"{_safe(a, f'precision@{k}'):>8}  "
            f"{_safe(a, f'recall@{k}'):>7}  "
            f"{_safe(a, f'mrr@{k}'):>7}  "
            f"{str(a.get('n_evaluated','?')):>5}"
        )
    tbl6b.append(f"{'='*80}")

    # ── TABLE VII — Answer Quality ───────────────────────────────────────────
    tbl7 = []
    tbl7.append(f"\n{'='*104}")
    tbl7.append("TABLE VII — Answer Quality (Held-Out Test Primary; Combined Supplementary)")
    tbl7.append(f"{'='*104}")
    hdr7 = (
        f"{'Retrieval Strategy':<30} {'Corr.T':>8} {'Corr.All':>9} {'Ground.T':>9} "
        f"{'Relev.T':>8} {'RetRel.T':>9} {'N Test':>7} {'N All':>6}"
    )
    tbl7.append(hdr7)
    tbl7.append("-"*104)

    for mode in RETRIEVERS:
        lbl = RETRIEVER_LABELS.get(mode, mode)
        rep = _load_json(_RESULTS_DIR / f"{mode}_report.json") or {}
        n = None
        n_all = None

        def get_val(metric_name):
            m_json = _load_json(_RESULTS_DIR / f"{mode}_{metric_name}.json")
            if m_json:
                return (
                    m_json.get('average_score'),
                    m_json.get('total_evaluated'),
                    m_json.get('combined_average_score'),
                    m_json.get('combined_total_evaluated'),
                )
            if metric_name in rep:
                return rep[metric_name], rep.get("total_questions"), None, None
            return None, None, None, None

        corr, n_c, corr_all, n_all = get_val('correctness')
        grou, n_g, _, _ = get_val('groundedness')
        rele, n_r, _, _ = get_val('relevance')
        retr, n_rr, _, _ = get_val('retrieval_relevance')

        for nx in (n_c, n_g, n_r, n_rr):
            if nx and not n:
                n = nx
        if not n:
            n = "?"
        if not n_all:
            n_all = "?"

        tbl7.append(
            f"{lbl:<30} "
            f"{_safe({'v': corr}, 'v'):>8} "
            f"{_safe({'v': corr_all}, 'v'):>9} "
            f"{_safe({'v': grou}, 'v'):>9} "
            f"{_safe({'v': rele}, 'v'):>8} "
            f"{_safe({'v': retr}, 'v'):>9} "
            f"{str(n):>7} "
            f"{str(n_all):>6}"
        )
    tbl7.append(f"{'='*104}")

    tbl8 = []
    tbl8.append(f"\n{'='*76}")
    tbl8.append("TABLE VIII — Out-of-Scope Refusal Compliance")
    tbl8.append(f"{'='*76}")
    tbl8.append(f"{'Retrieval Strategy':<34} {'Test rate':>10} {'All rate':>10} {'N Test':>8} {'N All':>7}")
    tbl8.append("-"*76)
    for mode in RETRIEVERS:
        lbl = RETRIEVER_LABELS.get(mode, mode)
        payload = _load_json(_RESULTS_DIR / f"{mode}_retrieval_metrics.json") or {}
        refusal = payload.get("refusal", {})
        test_refusal = refusal.get("by_split", {}).get("test", {})
        tbl8.append(
            f"{lbl:<34} "
            f"{_safe(test_refusal, 'refusal_compliance'):>10} "
            f"{_safe(refusal, 'refusal_compliance'):>10} "
            f"{str(test_refusal.get('n','?')):>8} "
            f"{str(refusal.get('n','?')):>7}"
        )
    tbl8.append(f"{'='*76}")

    return tbl6, tbl6b, tbl7, tbl8


def main():
    tbl6, tbl6b, tbl7, tbl8 = build_tables()
    all_lines = tbl6 + tbl6b + tbl7 + tbl8

    output_txt = _THIS_DIR / "ieee_tables.txt"
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_lines) + '\n')

    for line in all_lines:
        print(line)

    machine = {
        "table_VI_retrieval_strategy_comparison": tbl6,
        "table_VIB_combined_retrieval_supplementary": tbl6b,
        "table_VII_answer_quality": tbl7,
        "table_VIII_refusal_compliance": tbl8,
    }
    output_json = _THIS_DIR / "ieee_tables.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(machine, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Tables saved:")
    print(f"   {output_txt}")
    print(f"   {output_json}")
    print("\n✅ Retrieval ablation results formatted for IEEE paper!")
    print("   Copy Tables VI and VII from ieee_tables.txt into the LaTeX source.\n")


if __name__ == "__main__":
    main()
