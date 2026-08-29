#!/usr/bin/env python
"""
Step 1 — Verify Hybrid Index (Retrieval Ablation)
==================================================
Verifies that the hybrid legal-first index (built by the chunking
ablation) is present and valid before running the retrieval ablation.

The retrieval ablation uses the SAME hybrid index for all three
retrieval strategies:
  • faiss_only  — dense/semantic retrieval only
  • bm25_only   — pure BM25 only (no dafa boost, no semantic)
  • hybrid      — full dafa-boost + BM25 + semantic (deployed)

Usage:
  python 01_verify_hybrid_index.py
  python 01_verify_hybrid_index.py --index-dir ../chunking_ablation/indexes/hybrid
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

_DEFAULT_INDEX = _EXPERIMENTS_DIR / "chunking_ablation" / "indexes" / "hybrid"

REQUIRED_FILES = [
    "chunks.json",
    "faiss.index",
    "faiss_bm25.pkl",
    "faiss_manifest.json",
]


def verify_index(index_dir: str) -> bool:
    idx = Path(index_dir)

    print(f"\n{'='*65}")
    print(f"  Verifying hybrid index at: {idx}")
    print(f"{'='*65}")

    if not idx.exists():
        print(f"❌ Index directory does not exist: {idx}")
        print("   → Run chunking_ablation/01_build_all_indexes.py first")
        return False

    all_ok = True
    for fname in REQUIRED_FILES:
        fpath = idx / fname
        exists = fpath.exists()
        size   = fpath.stat().st_size if exists else 0
        status = "✅" if exists else "❌"
        print(f"  {status} {fname:<25} {size:>10,d} bytes")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\n❌ Index is incomplete. Rebuild with:")
        print("   python chunking_ablation/01_build_all_indexes.py --strategy hybrid")
        return False

    # Read manifest
    manifest_path = idx / "faiss_manifest.json"
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    print(f"\n📋 Manifest details:")
    print(f"   Built at   : {manifest.get('built_at', '?')}")
    print(f"   Model      : {manifest.get('model_name', '?')}")
    print(f"   Chunk count: {manifest.get('chunk_count', '?')}")
    for src in manifest.get('source_files', []):
        print(f"   Source     : {src.get('path','?')}  ({src.get('size_bytes',0):,} bytes)")

    # Quick stats
    stats_path = idx / "index_stats.json"
    if stats_path.exists():
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        print(f"\n📊 Index statistics:")
        print(f"   Chunks   : {stats.get('chunk_count')}")
        print(f"   Avg chars: {stats.get('avg_chars')}")
        print(f"   Min chars: {stats.get('min_chars')}")
        print(f"   Max chars: {stats.get('max_chars')}")

    print(f"\n✅ Hybrid index is valid and ready for retrieval ablation!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Verify hybrid index is ready for retrieval ablation"
    )
    parser.add_argument(
        "--index-dir",
        default=str(_DEFAULT_INDEX),
        help=f"Path to hybrid index directory (default: {_DEFAULT_INDEX})",
    )
    args = parser.parse_args()

    print("\n╔" + "═"*63 + "╗")
    print("║" + " "*10 + "LawBot IEEE — Verify Hybrid Index" + " "*20 + "║")
    print("╚" + "═"*63 + "╝")

    ok = verify_index(args.index_dir)

    if ok:
        print("\n  Next step: python 02_populate_answers.py")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
