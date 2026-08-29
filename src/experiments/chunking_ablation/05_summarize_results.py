#!/usr/bin/env python
"""
Step 5 — Summarize & Format Results (Chunking Ablation)
=======================================================
Reads all generated result files and produces IEEE-formatted tables
for inclusion in the paper.

Tables generated:
  TABLE II  — Chunk Statistics (count, avg chars, min, max)
  TABLE IV  — Answer Quality (correctness, groundedness, relevance,
               retrieval_relevance, overall)
  TABLE V   — Retrieval Metrics (Hit@1, Hit@5, P@5, R@5, MRR@5)

Output:
  experiments/chunking_ablation/ieee_tables.txt   (plain text for copy-paste)
  experiments/chunking_ablation/ieee_tables.json  (machine-readable)

Usage:
  python 05_summarize_results.py
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="strict", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="strict", line_buffering=True)
    os.environ["PYTHONIOENCODING"] = "utf-8"

_THIS_DIR    = Path(__file__).resolve().parent
_RESULTS_DIR = _THIS_DIR / "results"

STRATEGIES = ["recursive", "dafa", "semantic", "hybrid"]

STRATEGY_LABELS = {
    "recursive": "Recursive Char.",
    "dafa":      "Dafa-Based",
    "semantic":  "Semantic",
    "hybrid":    "Hybrid (ours)",
}


def _load_json(path: Path) -> Optional[Dict]:
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def _safe(d: dict, *keys, default="—") -> str:
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
    # ── TABLE II — Chunk Statistics ──────────────────────────────────────────
    index_summary = _load_json(_THIS_DIR / "index_summary.json") or {}

    tbl2 = []
    tbl2.append(f"\n{'='*70}")
    tbl2.append("TABLE II — Chunking Strategy Statistics")
    tbl2.append(f"{'='*70}")
    hdr = f"{'Strategy':<18} {'Chunks':>8} {'Avg Chars':>10} {'Min':>7} {'Max':>7} {'Merged%':>8}"
    tbl2.append(hdr)
    tbl2.append("-"*70)
    for s in STRATEGIES:
        stats = index_summary.get(s, {})
        if not stats or stats.get("skipped"):
            tbl2.append(f"{STRATEGY_LABELS[s]:<18} {'N/A':>8}")
            continue
        n     = stats.get('chunk_count', 0)
        multi = stats.get('multi_dafa', 0)
        pct   = f"{100*multi/n:.1f}%" if n else "—"
        tbl2.append(
            f"{STRATEGY_LABELS[s]:<18} {n:>8} "
            f"{stats.get('avg_chars',0):>10.1f} "
            f"{stats.get('min_chars',0):>7} "
            f"{stats.get('max_chars',0):>7} "
            f"{pct:>8}"
        )
    tbl2.append(f"{'='*70}")

    # ── TABLE IV — Answer Quality ────────────────────────────────────────────
    tbl4 = []
    tbl4.append(f"\n{'='*96}")
    tbl4.append("TABLE IV — Answer Quality (Held-Out Test Primary; Combined Supplementary)")
    tbl4.append(f"{'='*96}")
    hdr4 = (
        f"{'Strategy':<18} {'Corr.T':>8} {'Corr.All':>9} {'Ground.T':>9} "
        f"{'Relev.T':>8} {'RetRel.T':>9} {'N Test':>7} {'N All':>6}"
    )
    tbl4.append(hdr4)
    tbl4.append("-"*96)

    for s in STRATEGIES:
        rep = _load_json(_RESULTS_DIR / f"{s}_report.json") or {}
        n = None
        n_all = None

        def get_val(metric_name):
            m_json = _load_json(_RESULTS_DIR / f"{s}_{metric_name}.json")
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

        tbl4.append(
            f"{STRATEGY_LABELS[s]:<18} "
            f"{_safe({'v': corr}, 'v'):>8} "
            f"{_safe({'v': corr_all}, 'v'):>9} "
            f"{_safe({'v': grou}, 'v'):>9} "
            f"{_safe({'v': rele}, 'v'):>8} "
            f"{_safe({'v': retr}, 'v'):>9} "
            f"{str(n):>7} "
            f"{str(n_all):>6}"
        )
    tbl4.append(f"{'='*96}")

    # ── TABLE V — Retrieval Metrics ──────────────────────────────────────────
    ret_summary = _load_json(_THIS_DIR / "retrieval_metrics_summary.json") or {}

    tbl5 = []
    tbl5.append(f"\n{'='*80}")
    tbl5.append("TABLE V — Held-Out Test Retrieval Metrics (Chunking Ablation, k=5)")
    tbl5.append(f"{'='*80}")
    hdr5 = (
        f"{'Strategy':<18} {'Hit@1':>7} {'Hit@5':>7} "
        f"{'Prec@5':>8} {'Rec@5':>7} {'MRR@5':>7} {'N':>5}"
    )
    tbl5.append(hdr5)
    tbl5.append("-"*80)

    for s in STRATEGIES:
        a = ret_summary.get(s, {})
        if not a:
            m_json = _load_json(_RESULTS_DIR / f"{s}_retrieval_metrics.json")
            if m_json:
                a = m_json.get("aggregate", {})

        if "error" in a:
            tbl5.append(f"{STRATEGY_LABELS[s]:<18}  (error)")
            continue
        tbl5.append(
            f"{STRATEGY_LABELS[s]:<18} "
            f"{_safe(a, 'hit@1'):>7} "
            f"{_safe(a, 'hit@5'):>7} "
            f"{_safe(a, 'precision@5'):>8} "
            f"{_safe(a, 'recall@5'):>7} "
            f"{_safe(a, 'mrr@5'):>7} "
            f"{str(a.get('n_evaluated','?')):>5}"
        )
    tbl5.append(f"{'='*80}")

    tbl5b = []
    tbl5b.append(f"\n{'='*80}")
    tbl5b.append("TABLE V-B — Combined 350-Question Retrieval Metrics (Supplementary)")
    tbl5b.append(f"{'='*80}")
    tbl5b.append(hdr5)
    tbl5b.append("-"*80)
    for s in STRATEGIES:
        payload = _load_json(_RESULTS_DIR / f"{s}_retrieval_metrics.json") or {}
        a = payload.get("aggregate_combined", {})
        tbl5b.append(
            f"{STRATEGY_LABELS[s]:<18} "
            f"{_safe(a, 'hit@1'):>7} "
            f"{_safe(a, 'hit@5'):>7} "
            f"{_safe(a, 'precision@5'):>8} "
            f"{_safe(a, 'recall@5'):>7} "
            f"{_safe(a, 'mrr@5'):>7} "
            f"{str(a.get('n_evaluated','?')):>5}"
        )
    tbl5b.append(f"{'='*80}")

    tbl5c = []
    tbl5c.append(f"\n{'='*68}")
    tbl5c.append("TABLE V-C — Out-of-Scope Refusal Compliance")
    tbl5c.append(f"{'='*68}")
    tbl5c.append(f"{'Strategy':<22} {'Test rate':>10} {'All rate':>10} {'N Test':>8} {'N All':>7}")
    tbl5c.append("-"*68)
    for s in STRATEGIES:
        payload = _load_json(_RESULTS_DIR / f"{s}_retrieval_metrics.json") or {}
        refusal = payload.get("refusal", {})
        test_refusal = refusal.get("by_split", {}).get("test", {})
        tbl5c.append(
            f"{STRATEGY_LABELS[s]:<22} "
            f"{_safe(test_refusal, 'refusal_compliance'):>10} "
            f"{_safe(refusal, 'refusal_compliance'):>10} "
            f"{str(test_refusal.get('n','?')):>8} "
            f"{str(refusal.get('n','?')):>7}"
        )
    tbl5c.append(f"{'='*68}")

    return tbl2, tbl4, tbl5, tbl5b, tbl5c


def main():
    tbl2, tbl4, tbl5, tbl5b, tbl5c = build_tables()
    all_lines = tbl2 + tbl4 + tbl5 + tbl5b + tbl5c

    output_txt = _THIS_DIR / "ieee_tables.txt"
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_lines) + '\n')

    for line in all_lines:
        print(line)

    # Machine-readable version
    machine = {
        "table_II_chunk_stats":  tbl2,
        "table_IV_answer_quality": tbl4,
        "table_V_retrieval_metrics": tbl5,
        "table_VB_combined_retrieval_supplementary": tbl5b,
        "table_VC_refusal_compliance": tbl5c,
    }
    output_json = _THIS_DIR / "ieee_tables.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(machine, f, ensure_ascii=False, indent=2)

    print(f"\n\n💾 Tables saved:")
    print(f"   {output_txt}")
    print(f"   {output_json}")
    print("\n✅ Chunking ablation results formatted for IEEE paper!")
    print("   Copy tables from ieee_tables.txt into the LaTeX source.\n")


if __name__ == "__main__":
    main()
