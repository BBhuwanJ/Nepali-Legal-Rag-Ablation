#!/usr/bin/env python
"""
Step 2 — Populate Answers (Chunking Ablation)
==============================================
Runs the RAG pipeline over each strategy-specific chunk index using the
ROUTED HYBRID retriever (held constant across all strategies) and generates
answers for every question in the evaluation dataset.

Output:
  experiments/chunking_ablation/results/<strategy>_populated.json

Usage:
  # Populate all four strategies
  python 02_populate_answers.py

  # Populate a single strategy
  python 02_populate_answers.py --strategy recursive

  # Dry-run: process only 3 questions per strategy (for testing)
  python 02_populate_answers.py --strategy recursive --dry-run 3

  # Override top-k
  python 02_populate_answers.py --top-k 5
"""
from __future__ import annotations

import os
import sys
import asyncio
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

from shared.populate_utils import populate  # type: ignore[import]

_INDEXES_DIR = _THIS_DIR / "indexes"
_RESULTS_DIR = _THIS_DIR / "results"
_EVAL_DATA   = _EVAL_V2_DIR / "evalData_gold_v2.json"

STRATEGIES = ["recursive", "dafa", "semantic", "hybrid"]

# ── Constant retriever for the chunking ablation ─────────────────────────────
RETRIEVAL_MODE = "routed_hybrid"


async def populate_strategy(
    strategy: str,
    top_k: int,
    dry_run: int,
    restart: bool,
    eval_data: Path,
) -> None:
    index_dir = str(_INDEXES_DIR / strategy)
    if not (Path(index_dir) / "faiss_manifest.json").exists():
        print(f"❌ Index not found for [{strategy}]. Run 01_build_all_indexes.py first.")
        return

    out_path = str(_RESULTS_DIR / f"{strategy}_populated.json")
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Populating [{strategy}] → {out_path}")
    print(f"  Retrieval mode : {RETRIEVAL_MODE}  (constant)")
    print(f"  Top-k          : {top_k}")
    print(f"{'='*65}")

    await populate(
        eval_data_path=str(eval_data),
        output_path=out_path,
        index_dir=index_dir,
        retrieval_mode=RETRIEVAL_MODE,
        top_k=top_k,
        skip_oos=False,
        dry_run=dry_run,
        restart=restart,
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Populate eval data for chunking ablation (routed hybrid retriever constant)"
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES + ["all"],
        default="all",
        help="Which strategy to populate (default: all)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of chunks to retrieve per question (default: 5)",
    )
    parser.add_argument(
        "--dry-run", type=int, default=0, metavar="N",
        help="Only process N questions per strategy (0 = all)",
    )
    parser.add_argument(
        "--restart", action="store_true",
        help="Replace populated outputs instead of resuming them",
    )
    parser.add_argument(
        "--dataset", type=Path, default=_EVAL_DATA,
        help="Evaluation dataset (default: frozen 350-record gold benchmark)",
    )
    args = parser.parse_args()

    to_run = STRATEGIES if args.strategy == "all" else [args.strategy]

    print("\n╔" + "═"*63 + "╗")
    print("║" + " "*10 + "LawBot IEEE — Populate Answers (Chunking)" + " "*12 + "║")
    print("╚" + "═"*63 + "╝")
    print(f"\n  Strategies     : {to_run}")
    print(f"  Retriever      : {RETRIEVAL_MODE}  ← CONSTANT")
    print(f"  Top-k          : {args.top_k}")
    print(f"  Dry-run limit  : {args.dry_run or 'disabled'}")
    print(f"  Eval dataset   : {args.dataset}")

    for strategy in to_run:
        await populate_strategy(
            strategy, args.top_k, args.dry_run, args.restart, args.dataset
        )

    print("\n✅ Population complete! Next step: run 03_evaluate_answers.py")


if __name__ == "__main__":
    asyncio.run(main())
