"""
Hybrid Legal-First Semantic Chunker — Chunking Strategy 4
===========================================================
Thin wrapper around the deployed production chunker
(app.rag.preprocessor.load_and_chunk_document).

This is the CONSTANT chunk index used in the retrieval ablation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict

# ── Resolve project root ────────────────────────────────────────────────────
_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _EXPERIMENTS_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.rag.preprocessor import load_and_chunk_document


def chunk_document_hybrid(
    file_path: str,
    source: str = "muluki_ain",
    act_name: str = "मुलुकी देवानी (संहिता) ऐन, २०७४",
    use_semantic: bool = True,
) -> List[Dict]:
    """
    Chunk a Nepali legal document using the deployed hybrid legal-first
    semantic strategy.

    Delegates entirely to the production preprocessor so that results
    are byte-for-byte identical to the deployed index.

    Args:
        file_path    : Path to the source .txt document.
        source       : Source key, e.g. "muluki_ain".
        act_name     : Display name of the act (Nepali).
        use_semantic : Enable semantic merging (default True, same as production).

    Returns:
        List of chunk dicts compatible with the FAISS/BM25 index schema.
    """
    print(f"📖 [hybrid] Loading: {file_path}")
    chunks = load_and_chunk_document(
        file_path,
        use_semantic=use_semantic,
        source=source,
        act_name=act_name,
    )
    print(f"✅ [hybrid] {len(chunks)} chunks produced by legal-first strategy")
    return chunks
