#!/usr/bin/env python
"""
Step 1 — Build All Strategy-Specific Indexes
=============================================
Chunks both legal documents using each of the four strategies and
builds a FAISS + BM25 index for each.

Output layout:
  experiments/chunking_ablation/indexes/
    recursive/   — fixed-size recursive character splitter
    dafa/        — pure dafa-based (no semantic merging)
    semantic/    — semantic similarity grouping
    hybrid/      — hybrid legal-first semantic (deployed strategy)

Usage:
  # Build all four indexes
  python 01_build_all_indexes.py

  # Build only one strategy
  python 01_build_all_indexes.py --strategy recursive

  # Force rebuild even if index exists
  python 01_build_all_indexes.py --force
"""
from __future__ import annotations

import os
import sys
import json
import hashlib
import argparse
import warnings
import re
from pathlib import Path

# ── Suppress noisy TF/Keras warnings ────────────────────────────────────────
os.environ["TF_ENABLE_ONEDNN_OPTS"]          = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]           = "3"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]         = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Windows UTF-8 fix (line-buffered for immediate terminal output) ───────────
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="strict", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="strict", line_buffering=True)
    os.environ["PYTHONIOENCODING"] = "utf-8"

# ── Resolve paths ─────────────────────────────────────────────────────────────
_THIS_DIR        = Path(__file__).resolve().parent          # chunking_ablation/
_EXPERIMENTS_DIR = _THIS_DIR.parent                         # experiments/
_BACKEND_DIR     = _EXPERIMENTS_DIR.parent                  # backend/
_SHARED_DIR      = _EXPERIMENTS_DIR / "shared"
_INDEXES_DIR     = _THIS_DIR / "indexes"

for _p in [str(_BACKEND_DIR), str(_EXPERIMENTS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.chunkers.recursive_chunker import chunk_document_recursive
from shared.chunkers.dafa_chunker       import chunk_document_dafa
from shared.chunkers.semantic_chunker   import chunk_document_semantic
from shared.chunkers.hybrid_chunker     import chunk_document_hybrid
from shared.corpus                      import DOCUMENTS, require_admitted_corpus
from shared.index_builder               import build_and_save_index


# ── Document registry ─────────────────────────────────────────────────────────
def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_matches_documents(marker: str, source_files: list[str]) -> bool:
    """Return True only when an existing index covers the current corpus."""
    try:
        with open(marker, "r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        recorded = {
            os.path.normcase(os.path.abspath(entry["path"])): entry.get("sha256")
            for entry in manifest.get("source_files", [])
        }
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False

    expected = {
        os.path.normcase(os.path.abspath(path)): _sha256(path)
        for path in source_files
    }
    return recorded == expected


# ── Strategy definitions ──────────────────────────────────────────────────────
def _chunk_recursive(docs) -> list:
    all_chunks = []
    for path, source, act_name in docs:
        if not os.path.exists(path):
            print(f"⚠️  Not found: {path}")
            continue
        chunks = chunk_document_recursive(
            path, source=source, act_name=act_name,
            chunk_size=900, chunk_overlap=150,
        )
        all_chunks.extend(chunks)
    return all_chunks


def _chunk_dafa(docs) -> list:
    all_chunks = []
    for path, source, act_name in docs:
        if not os.path.exists(path):
            print(f"⚠️  Not found: {path}")
            continue
        chunks = chunk_document_dafa(path, source=source, act_name=act_name)
        all_chunks.extend(chunks)
    return all_chunks


def _chunk_semantic(docs) -> list:
    all_chunks = []
    for path, source, act_name in docs:
        if not os.path.exists(path):
            print(f"⚠️  Not found: {path}")
            continue
        chunks = chunk_document_semantic(
            path, source=source, act_name=act_name,
            sim_threshold=0.75, max_chars=900,
        )
        all_chunks.extend(chunks)
    return all_chunks


def _chunk_hybrid(docs) -> list:
    all_chunks = []
    for path, source, act_name in docs:
        if not os.path.exists(path):
            print(f"⚠️  Not found: {path}")
            continue
        chunks = chunk_document_hybrid(path, source=source, act_name=act_name)
        all_chunks.extend(chunks)
    return all_chunks


STRATEGIES = {
    "recursive": {
        "label":       "Recursive Character Splitter",
        "chunker":     _chunk_recursive,
        "out_subdir":  "recursive",
    },
    "dafa": {
        "label":       "Dafa-Based Chunker",
        "chunker":     _chunk_dafa,
        "out_subdir":  "dafa",
    },
    "semantic": {
        "label":       "Semantic Chunker",
        "chunker":     _chunk_semantic,
        "out_subdir":  "semantic",
    },
    "hybrid": {
        "label":       "Hybrid Legal-First Semantic Chunker",
        "chunker":     _chunk_hybrid,
        "out_subdir":  "hybrid",
    },
}


_STANDALONE_PROVISION = re.compile(r'^\s*[०-९\d]+(?:[क-हA-Za-z])?[.)]?\s*$')


def _validate_chunk_output(name: str, chunks: list[dict]) -> None:
    """Fail closed on chunk artifacts that invalidate an ablation."""
    empty = [chunk for chunk in chunks if not str(chunk.get("text", "")).strip()]
    if empty:
        raise RuntimeError(f"{name} produced {len(empty)} empty chunks")

    if name == "semantic":
        standalone = [
            chunk for chunk in chunks
            if _STANDALONE_PROVISION.fullmatch(str(chunk.get("text", "")))
        ]
        too_small = [
            chunk for chunk in chunks
            if len(str(chunk.get("text", "")).strip()) < 100
        ]
        if standalone or too_small:
            raise RuntimeError(
                "Semantic chunk quality gate failed: "
                f"standalone_provisions={len(standalone)}, "
                f"chunks_under_100_chars={len(too_small)}"
            )


def build_strategy(name: str, force: bool = False) -> dict:
    """Build index for a single strategy. Returns stats dict."""
    cfg     = STRATEGIES[name]
    out_dir = str(_INDEXES_DIR / cfg["out_subdir"])
    marker  = os.path.join(out_dir, "faiss_manifest.json")

    print(f"\n{'='*65}")
    print(f"  Strategy : {cfg['label']} [{name}]")
    print(f"  Output   : {out_dir}")
    print(f"{'='*65}")

    source_files = [p for p, _, _ in DOCUMENTS]
    missing_files = [p for p in source_files if not os.path.exists(p)]
    if missing_files:
        missing = "\n  - ".join(missing_files)
        raise FileNotFoundError(
            f"The registered experimental corpus is incomplete:\n  - {missing}"
        )

    if (os.path.exists(marker) and not force
            and _manifest_matches_documents(marker, source_files)):
        print("⏭️  Index already exists. Use --force to rebuild.\n")
        stats_path = os.path.join(out_dir, "index_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                return json.load(f)
        return {"strategy": name, "skipped": True}
    if os.path.exists(marker) and not force:
        print("♻️  Existing index is stale for the registered corpus; rebuilding.")

    # Chunk documents
    print(f"\n📋 STEP 1: Chunking with [{name}]…")
    all_chunks = cfg["chunker"](DOCUMENTS)

    if not all_chunks:
        print(f"❌ No chunks produced for {name}. Skipping.")
        return {"strategy": name, "error": "no_chunks"}

    _validate_chunk_output(name, all_chunks)

    # Re-assign sequential IDs across documents
    for idx, chunk in enumerate(all_chunks, start=1):
        chunk["chunk_id"] = idx

    print(f"\n📊 Total combined chunks: {len(all_chunks)}")

    # Build FAISS + BM25 index
    print(f"\n📋 STEP 2: Building index…")
    stats = build_and_save_index(all_chunks, out_dir, source_files=source_files)
    stats["strategy"] = name
    stats["label"]    = cfg["label"]

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build chunking-strategy indexes for the IEEE ablation"
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()) + ["all"],
        default="all",
        help="Which strategy to build (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild even if the index already exists",
    )
    args = parser.parse_args()

    # Legal corpus quality is a hard precondition, not a warning. Embedding a
    # known-corrupt statute would make every downstream metric misleading.
    require_admitted_corpus()

    to_build = list(STRATEGIES.keys()) if args.strategy == "all" else [args.strategy]

    print("\n" + "╔" + "═"*63 + "╗")
    print("║" + " "*15 + "LawBot IEEE — Build All Indexes" + " "*17 + "║")
    print("╚" + "═"*63 + "╝\n")
    print(f"  Strategies to build : {to_build}")
    print(f"  Output directory    : {_INDEXES_DIR}")
    print(f"  Force rebuild       : {args.force}")

    _INDEXES_DIR.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for name in to_build:
        stats = build_strategy(name, force=args.force)
        all_stats[name] = stats

    # Always report all current strategy artifacts. A targeted rebuild must not
    # erase the combined summary or make the paper appear to have one strategy.
    summary_stats = {}
    source_files = [path for path, _, _ in DOCUMENTS]
    for name, cfg in STRATEGIES.items():
        if name in all_stats:
            summary_stats[name] = all_stats[name]
            continue
        out_dir = _INDEXES_DIR / cfg["out_subdir"]
        marker = out_dir / "faiss_manifest.json"
        stats_path = out_dir / "index_stats.json"
        if stats_path.is_file() and _manifest_matches_documents(str(marker), source_files):
            with open(stats_path, "r", encoding="utf-8") as stats_file:
                stats = json.load(stats_file)
            stats["strategy"] = name
            stats["label"] = cfg["label"]
            summary_stats[name] = stats
        else:
            summary_stats[name] = {"strategy": name, "error": "missing_or_stale_index"}

    # Summary table
    print("\n\n" + "="*65)
    print("📊  CHUNKING STRATEGY INDEX SUMMARY")
    print("="*65)
    header = f"{'Strategy':<12} {'Chunks':>7} {'Avg chars':>10} {'Min':>6} {'Max':>6} {'Multi':>6}"
    print(header)
    print("-"*65)
    for name, s in summary_stats.items():
        if s.get("skipped") or s.get("error"):
            print(f"{name:<12}  (skipped/error)")
            continue
        print(
            f"{name:<12} {s.get('chunk_count',0):>7} "
            f"{s.get('avg_chars',0):>10.1f} "
            f"{s.get('min_chars',0):>6} "
            f"{s.get('max_chars',0):>6} "
            f"{s.get('multi_dafa',0):>6}"
        )
    print("="*65)

    # Save combined summary
    summary_path = _THIS_DIR / "index_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Summary saved: {summary_path}")
    print("\n✅ Done! Next step: run 02_populate_answers.py")


if __name__ == "__main__":
    main()
