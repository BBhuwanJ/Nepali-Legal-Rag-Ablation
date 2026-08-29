"""
Shared Index Builder
=====================
Builds a FAISS + BM25 index from a list of chunks and saves all
artifacts to a given output directory.

Output layout:
  <out_dir>/
    chunks.json        — serialised chunk list
    faiss.index        — FAISS IndexFlatIP
    faiss_bm25.pkl     — pickled BM25 inverted-index cache
    faiss_manifest.json — build metadata (timestamp, model, sha256s)
"""
from __future__ import annotations

import sys
import os
import json
import pickle
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np

# ── Resolve backend dir ──────────────────────────────────────────────────────
_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _EXPERIMENTS_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.rag.vector_store_improved import HybridRetriever


_MODEL_NAME = "Yunika/sentence-transformer-nepali"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"🔄 Loading embedding model ({_MODEL_NAME})…")
        _model = SentenceTransformer(_MODEL_NAME)
        print("✅ Embedding model ready")
    return _model


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def _atomic_write_bytes(path: str, data: bytes) -> None:
    dir_ = os.path.dirname(path) or '.'
    with tempfile.NamedTemporaryFile('wb', dir=dir_, delete=False, suffix='.tmp') as tf:
        tf.write(data)
        tmp = tf.name
    os.replace(tmp, path)


def _atomic_write_text(path: str, data: str) -> None:
    dir_ = os.path.dirname(path) or '.'
    with tempfile.NamedTemporaryFile('w', dir=dir_, delete=False,
                                     suffix='.tmp', encoding='utf-8') as tf:
        tf.write(data)
        tmp = tf.name
    os.replace(tmp, path)


def build_and_save_index(
    chunks: List[Dict],
    out_dir: str,
    source_files: Optional[List[str]] = None,
) -> Dict:
    """
    Build FAISS + BM25 index from chunks and persist to *out_dir*.

    Args:
        chunks       : Chunk list produced by any of the chunkers.
        out_dir      : Target directory (created if absent).
        source_files : Original .txt paths recorded in manifest SHA-256s.

    Returns:
        A dict with summary statistics (chunk_count, avg_chars, …).
    """
    os.makedirs(out_dir, exist_ok=True)
    out = Path(out_dir)

    print(f"\n🔨 Building index → {out_dir}")
    print(f"   Chunks: {len(chunks)}")

    # ── 1. Embeddings ────────────────────────────────────────────────────────
    model = _get_model()
    texts = [c['text'] for c in chunks]
    print("🔄 Generating embeddings…")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    ).astype('float32')

    # ── 2. FAISS index ───────────────────────────────────────────────────────
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss_path = str(out / 'faiss.index')
    tmp_faiss  = faiss_path + '.tmp'
    faiss.write_index(index, tmp_faiss)
    os.replace(tmp_faiss, faiss_path)
    print(f"✅ FAISS index saved ({index.ntotal} vectors, dim={dim})")

    # ── 3. Chunks JSON ───────────────────────────────────────────────────────
    chunks_path = str(out / 'chunks.json')
    _atomic_write_text(chunks_path, json.dumps(chunks, ensure_ascii=False, indent=2))
    print(f"✅ chunks.json saved")

    # ── 4. BM25 cache ────────────────────────────────────────────────────────
    print("🔄 Building BM25 index…")
    hybrid_retriever = HybridRetriever(chunks)
    bm25_cache = {
        'inverted_index': hybrid_retriever.inverted_index,
        'avg_doc_length': hybrid_retriever.avg_doc_length,
    }
    bm25_path = str(out / 'faiss_bm25.pkl')
    _atomic_write_bytes(bm25_path, pickle.dumps(bm25_cache))
    print("✅ BM25 cache saved")

    # ── 5. Manifest ──────────────────────────────────────────────────────────
    manifest = {
        'built_at':    datetime.utcnow().isoformat(),
        'model_name':  _MODEL_NAME,
        'chunk_count': len(chunks),
        'source_files': [
            {
                'path':   p,
                'sha256': _sha256(p),
                'size_bytes': os.path.getsize(p),
            }
            for p in (source_files or []) if os.path.exists(p)
        ],
    }
    manifest_path = str(out / 'faiss_manifest.json')
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
    print(f"✅ Manifest saved")

    # ── 6. Summary stats ─────────────────────────────────────────────────────
    lengths   = [c['char_count'] for c in chunks]
    multi_cnt = sum(1 for c in chunks if c.get('is_merged'))
    stats = {
        'out_dir':      out_dir,
        'chunk_count':  len(chunks),
        'avg_chars':    round(float(np.mean(lengths)), 2),
        'min_chars':    int(min(lengths)),
        'max_chars':    int(max(lengths)),
        'multi_dafa':   multi_cnt,
        'built_at':     manifest['built_at'],
    }
    stats_path = str(out / 'index_stats.json')
    _atomic_write_text(stats_path, json.dumps(stats, indent=2))
    print(f"\n📊 Index statistics:")
    for k, v in stats.items():
        print(f"   {k:20} : {v}")

    return stats
