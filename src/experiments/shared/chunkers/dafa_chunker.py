"""
Dafa-Based Chunker — Chunking Strategy 2
==========================================
Structure-aware chunking: each dafa (legal section) becomes exactly
one chunk.  Semantic merging is DISABLED, so neighbouring dafas are
never combined.  This isolates the "pure legal-boundary" baseline.

Reuses the dafa extraction logic from app.rag.preprocessor; the only
change is use_semantic=False.
"""
from __future__ import annotations

import re
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict

# ── Resolve project root so we can import app.rag ──────────────────────────
_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent  # .../experiments
_BACKEND_DIR = _EXPERIMENTS_DIR.parent                             # .../backend
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.rag.preprocessor import (
    extract_dafa_structure,
    validate_and_clean_dafas,
    finalize_chunks,
)


def chunk_document_dafa(
    file_path: str,
    source: str = "muluki_ain",
    act_name: str = "मुलुकी देवानी (संहिता) ऐन, २०७४",
) -> List[Dict]:
    """
    Chunk a Nepali legal document using pure dafa-based splitting.

    • Each dafa becomes its own chunk (no semantic merging).
    • Overlaps are still added at legal boundaries (same as deployed).
    • Chunks retain bhag/parichhed structural metadata.

    Args:
        file_path : Path to the source .txt document.
        source    : Source key, e.g. "muluki_ain".
        act_name  : Display name of the act (Nepali).

    Returns:
        List of chunk dicts compatible with the FAISS/BM25 index schema.
    """
    print(f"📖 [dafa] Loading: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Minimal normalisation (same as preprocessor.py)
    text = text.replace('\u200d', '').replace('\u200c', '')
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Step 1 — extract complete dafas with bhag/parichhed context
    dafas = extract_dafa_structure(text)

    # Step 2 — validate (marks incomplete ones but keeps them)
    validated = validate_and_clean_dafas(dafas)

    # Step 3 — NO semantic merging; wrap each dafa as a single-element group
    single_groups: List[Dict] = []
    for d in validated:
        single_groups.append({
            'dafa':       d['dafa'],
            'dafa_list':  [d['dafa']],
            'text':       d['text'],
            'is_merged':  False,
            'incomplete': d.get('incomplete', False),
            'bhag':       d.get('bhag', ''),
            'parichhed':  d.get('parichhed', ''),
        })

    # Step 4 — add overlaps & metadata via shared finalize_chunks
    chunks = finalize_chunks(
        single_groups,
        add_overlap=True,
        source=source,
        act_name=act_name,
    )

    _print_stats("dafa", chunks)
    return chunks


def _print_stats(strategy: str, chunks: List[Dict]) -> None:
    lengths = [c['char_count'] for c in chunks]
    multi   = sum(1 for c in chunks if c.get('is_merged'))
    print(f"\n{'='*55}")
    print(f"✅ [{strategy}] Chunking complete")
    print(f"   Total chunks  : {len(chunks)}")
    print(f"   Avg chars     : {np.mean(lengths):.1f}")
    print(f"   Min / Max     : {min(lengths)} / {max(lengths)}")
    print(f"   Multi-dafa    : {multi}")
    print(f"{'='*55}\n")
