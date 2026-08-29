#!/usr/bin/env python
"""
Step 2 — Populate Answers (Retrieval Ablation)
===============================================
Uses the HYBRID legal-first chunk index (constant) and generates
answers for every question under each of the three retrieval modes:

  faiss_only  — dense/semantic retrieval only (alpha=1.0, no BM25, no dafa boost)
  bm25_only   — pure BM25 only (alpha=0.0, no dafa boost, no semantic)
  hybrid      — dense + BM25 + Dafa fusion without inferred source routing
  routed_hybrid — proposed source-aware hybrid (same as the hybrid-chunk
                    entry in the chunking ablation)

Note: The 'routed_hybrid' mode here is IDENTICAL to the hybrid-chunk entry
in the chunking ablation. If that file already exists, this script reuses it
rather than regenerating answers.

Output:
  experiments/retrieval_ablation/results/<mode>_populated.json

Usage:
  python 02_populate_answers.py
  python 02_populate_answers.py --retriever faiss_only
  python 02_populate_answers.py --dry-run 3
"""
from __future__ import annotations

import os
import sys
import asyncio
import argparse
import shutil
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

from shared.populate_utils import populate, populated_matches_index  # type: ignore[import]

_HYBRID_INDEX  = _EXPERIMENTS_DIR / "chunking_ablation" / "indexes" / "hybrid"
_RESULTS_DIR   = _THIS_DIR / "results"
_EVAL_DATA     = _EVAL_V2_DIR / "evalData_gold_v2.json"

# The routed-hybrid populated file is shared with chunking ablation.
_CHUNKING_HYBRID_POPULATED = (
    _EXPERIMENTS_DIR / "chunking_ablation" / "results" / "hybrid_populated.json"
)

RETRIEVERS = ["faiss_only", "bm25_only", "hybrid", "routed_hybrid"]


async def populate_retriever(
    mode: str,
    top_k: int,
    dry_run: int,
    index_dir: str,
    restart: bool,
    eval_data: Path,
) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / f"{mode}_populated.json"

    # Reuse the hybrid-chunk + routed-retrieval reference configuration.
    if (mode == "routed_hybrid" and _CHUNKING_HYBRID_POPULATED.exists()
            and populated_matches_index(
                str(_CHUNKING_HYBRID_POPULATED), index_dir,
                eval_data_path=str(eval_data), retrieval_mode=mode, top_k=top_k,
            )):
        if (not out_path.exists() or restart
                or not populated_matches_index(
                    str(out_path), index_dir, eval_data_path=str(eval_data),
                    retrieval_mode=mode, top_k=top_k,
                )):
            shutil.copy(str(_CHUNKING_HYBRID_POPULATED), str(out_path))
            print(f"✅ [routed_hybrid] Reused from chunking ablation → {out_path}")
        else:
            print(f"⏭️  [routed_hybrid] Already exists: {out_path}")
        return

    print(f"\n{'='*65}")
    print(f"  Populating [{mode}] → {out_path}")
    print(f"  Index dir : {index_dir}")
    print(f"  Top-k     : {top_k}")
    print(f"{'='*65}")

    if not (Path(index_dir) / "faiss_manifest.json").exists():
        print(f"❌ Hybrid index not found. Run 01_verify_hybrid_index.py first.")
        return

    await populate(
        eval_data_path=str(eval_data),
        output_path=str(out_path),
        index_dir=index_dir,
        retrieval_mode=mode,
        top_k=top_k,
        skip_oos=False,
        dry_run=dry_run,
        restart=restart,
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Populate answers for retrieval ablation (hybrid chunk index constant)"
    )
    parser.add_argument(
        "--retriever",
        choices=RETRIEVERS + ["all"],
        default="all",
    )
    parser.add_argument("--top-k",   type=int, default=5)
    parser.add_argument("--dry-run", type=int, default=0, metavar="N")
    parser.add_argument(
        "--restart", action="store_true",
        help="Replace populated outputs instead of resuming them",
    )
    parser.add_argument(
        "--index-dir",
        default=str(_HYBRID_INDEX),
        help=f"Path to hybrid index (default: {_HYBRID_INDEX})",
    )
    parser.add_argument(
        "--dataset", type=Path, default=_EVAL_DATA,
        help="Evaluation dataset (default: frozen 350-record gold benchmark)",
    )
    args = parser.parse_args()

    to_run = RETRIEVERS if args.retriever == "all" else [args.retriever]

    print("\n╔" + "═"*63 + "╗")
    print("║" + " "*10 + "LawBot IEEE — Populate Answers (Retrieval)" + " "*11 + "║")
    print("╚" + "═"*63 + "╝")
    print(f"\n  Retrievers : {to_run}")
    print(f"  Index      : {args.index_dir}  ← CONSTANT (hybrid legal-first)")
    print(f"  Top-k      : {args.top_k}")
    print(f"  Dry-run    : {args.dry_run or 'disabled'}")
    print(f"  Dataset    : {args.dataset}")

    for mode in to_run:
        await populate_retriever(
            mode, args.top_k, args.dry_run, args.index_dir, args.restart,
            args.dataset,
        )

    print("\n✅ Population complete! Next step: run 04_retrieval_metrics.py")


if __name__ == "__main__":
    asyncio.run(main())
