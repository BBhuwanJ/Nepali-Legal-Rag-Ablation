#!/usr/bin/env python
"""
Step 3 — Evaluate Generated Answers (Retrieval Ablation)
=========================================================
Runs the 4 answer-quality metrics (correctness, groundedness, relevance,
retrieval_relevance) for each retrieval strategy's populated JSON file.

LLM Judge : gemini-3.5-flash  (temperature=0.0 for reproducibility)
Evaluation : All 4 metrics evaluated in one resumable pass by default.
             Gemini requests share a 5-RPM limiter (12s between starts).

Output:
  experiments/retrieval_ablation/results/<strategy>_<metric>.json
  experiments/retrieval_ablation/results/<strategy>_report.json

Usage:
  # Evaluate all strategies — combined mode (fastest, recommended)
  python 03_evaluate_answers.py

  # Evaluate one strategy
  python 03_evaluate_answers.py --strategy hybrid

  # Evaluate only specific metrics (uses sequential mode automatically)
  python 03_evaluate_answers.py --strategy hybrid --metric correctness groundedness

  # Force legacy sequential mode (for debugging)
  python 03_evaluate_answers.py --sequential
"""
from __future__ import annotations

import os
import sys
import argparse
import warnings
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"]          = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]           = "3"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]         = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="strict", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="strict", line_buffering=True)
    os.environ["PYTHONIOENCODING"] = "utf-8"

_THIS_DIR        = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent
_BACKEND_DIR     = _EXPERIMENTS_DIR.parent
_EVAL_V2_DIR     = _BACKEND_DIR / "evaluationV2"

for _p in [str(_BACKEND_DIR), str(_EXPERIMENTS_DIR), str(_EVAL_V2_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.evaluator import (  # type: ignore[import]
    run_correctness,
    run_groundedness,
    run_relevance,
    run_retrieval_relevance,
    run_all_metrics,
    run_all_metrics_sequential,
)

_RESULTS_DIR = _THIS_DIR / "results"
STRATEGIES   = ["faiss_only", "bm25_only", "hybrid", "routed_hybrid"]
ALL_METRICS  = ["correctness", "groundedness", "relevance", "retrieval_relevance"]

METRIC_FNS = {
    "correctness":         run_correctness,
    "groundedness":        run_groundedness,
    "relevance":           run_relevance,
    "retrieval_relevance": run_retrieval_relevance,
}


def _clear_outputs(strategy: str) -> None:
    for metric in ALL_METRICS:
        path = _RESULTS_DIR / f"{strategy}_{metric}.json"
        if path.exists():
            path.unlink()
    report = _RESULTS_DIR / f"{strategy}_report.json"
    if report.exists():
        report.unlink()


def evaluate_strategy(strategy: str, metrics: list, sequential: bool, restart: bool) -> None:
    populated = _RESULTS_DIR / f"{strategy}_populated.json"
    if not populated.exists():
        print(f"❌ Populated file not found for [{strategy}]. Run 02_populate_answers.py first.")
        return
    existing_outputs = [
        _RESULTS_DIR / f"{strategy}_{metric}.json" for metric in ALL_METRICS
    ]
    if restart:
        _clear_outputs(strategy)
    elif any(path.exists() and path.stat().st_mtime < populated.stat().st_mtime for path in existing_outputs):
        raise RuntimeError(
            f"[{strategy}] Existing evaluation outputs predate the populated file. "
            "Use --restart to prevent stale question-ID reuse."
        )

    print(f"\n{'='*65}")
    print(f"  Evaluating [{strategy}]")
    print(f"  Metrics : {metrics}")
    print(f"  Mode    : {'metric-by-metric' if sequential else 'combined single-pass (5 RPM)'}")
    print(f"{'='*65}")

    if set(metrics) == set(ALL_METRICS):
        # Full evaluation — use combined mode unless explicitly overridden
        if sequential:
            run_all_metrics_sequential(str(populated), str(_RESULTS_DIR), strategy_name=strategy)
        else:
            run_all_metrics(str(populated), str(_RESULTS_DIR), strategy_name=strategy)
    else:
        # Partial metric selection — always sequential
        for metric in metrics:
            out_file = str(_RESULTS_DIR / f"{strategy}_{metric}.json")
            print(f"\n▶ {metric.upper()}…")
            try:
                result = METRIC_FNS[metric](str(populated), out_file)
                print(f"   Score: {result.get('average_score', 'N/A')}")
            except Exception as e:
                print(f"   ❌ Failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated answers for retrieval ablation"
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES + ["all"],
        default="all",
    )
    parser.add_argument(
        "--metric",
        choices=ALL_METRICS,
        nargs="+",
        default=ALL_METRICS,
        help="Metrics to evaluate (default: all)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run one complete metric at a time instead of combined single-pass mode",
    )
    parser.add_argument(
        "--restart", action="store_true",
        help="Delete this strategy's old metric/report files before evaluation",
    )
    args = parser.parse_args()

    to_evaluate = STRATEGIES if args.strategy == "all" else [args.strategy]
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mode_label = "metric-by-metric" if args.sequential else "combined single-pass — shared 5-RPM limiter"

    print("\n╔" + "═"*63 + "╗")
    print("║" + " "*9 + "LawBot IEEE — Evaluate Answers (Retrieval)" + " "*12 + "║")
    print("╚" + "═"*63 + "╝")
    print(f"\n  Strategies : {to_evaluate}")
    print(f"  Metrics    : {args.metric}")
    print(f"  Mode       : {mode_label}")
    print(f"  Results dir: {_RESULTS_DIR}")
    print(f"  Judge model: gemini-3.5-flash  (temperature=0 — reproducible)")
    print("\n⚠️  This uses Gemini API calls. Ensure keys are in backend/.env")

    for strategy in to_evaluate:
        evaluate_strategy(strategy, args.metric, args.sequential, args.restart)

    print("\n✅ Evaluation complete! Next step: run 04_retrieval_metrics.py")


if __name__ == "__main__":
    main()
