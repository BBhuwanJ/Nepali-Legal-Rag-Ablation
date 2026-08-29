"""
Recursive Character Text Splitter — Chunking Strategy 1
=========================================================
Fixed-size splitting with character overlap.  No awareness of legal
(dafa) boundaries.  Serves as the baseline chunking strategy for the
IEEE paper ablation.

Parameters (matching the deployed run reported in the paper):
  chunk_size   = 900   characters
  chunk_overlap = 150  characters
"""
from __future__ import annotations

import re
import json
import os
import sys
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

# ── Devanagari helpers ──────────────────────────────────────────────────────
_DEV_TO_ARABIC = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
}

def _normalize_numbers(text: str) -> str:
    for d, a in _DEV_TO_ARABIC.items():
        text = text.replace(d, a)
    return text


# ── Core splitter ────────────────────────────────────────────────────────────
def _split_recursive(text: str,
                     separators: List[str],
                     chunk_size: int) -> List[str]:
    """Recursively split text by trying separators in order."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = separators[0] if separators else ""
    next_seps = separators[1:]

    parts = text.split(sep) if sep else list(text)

    chunks: List[str] = []
    current = ""

    for part in parts:
        joined = current + sep + part if current else part
        if len(joined) <= chunk_size:
            current = joined
        else:
            if current.strip():
                chunks.append(current)
            # If part itself is too long, recurse
            if len(part) > chunk_size and next_seps:
                sub = _split_recursive(part, next_seps, chunk_size)
                if sub:
                    chunks.extend(sub[:-1])
                    current = sub[-1]
                else:
                    current = part
            else:
                current = part

    if current.strip():
        chunks.append(current)

    return chunks


def _add_overlap(chunks: List[str], chunk_overlap: int) -> List[str]:
    """Prepend a suffix of the previous chunk as overlap."""
    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-chunk_overlap:]
        # Break at a clean boundary inside the overlap window
        for boundary in ['।', '\n', ' ']:
            idx = tail.rfind(boundary)
            if idx > 0:
                tail = tail[idx + 1:].strip()
                break
        combined = (tail + '\n' + chunks[i]) if tail else chunks[i]
        result.append(combined)
    return result


# ── Metadata extraction helpers ──────────────────────────────────────────────
_DAFA_IN_TEXT = re.compile(
    r'^([०-९\d]+(?:[क-हA-Za-z])?)(?:[.)]|\s+(?=[^\d\s]))\s*',
    re.MULTILINE,
)
_BHAG_HEADER     = re.compile(r'^भाग[–\-]?\s*([०-९\d]+)\s*(.*?)$', re.MULTILINE)
_PARICHHED_HDR   = re.compile(r'^परिच्छेद[–\-]?\s*([०-९\d]+)\s*(.*?)$', re.MULTILINE)


def _extract_dafas(chunk_text: str) -> List[str]:
    """Return dafa labels (raw, not normalized) found in chunk text."""
    seen: set = set()
    labels: List[str] = []
    for m in _DAFA_IN_TEXT.finditer(chunk_text):
        label = m.group(1)
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


# ── Public API ───────────────────────────────────────────────────────────────
def chunk_document_recursive(
    file_path: str,
    source: str = "muluki_ain",
    act_name: str = "मुलुकी देवानी (संहिता) ऐन, २०७४",
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> List[Dict]:
    """
    Chunk a Nepali legal document using recursive character splitting.

    Each chunk is annotated with best-effort dafa metadata extracted
    by regex from the chunk text.

    Args:
        file_path    : Path to the source .txt document.
        source       : Source key, e.g. "muluki_ain".
        act_name     : Display name of the act (Nepali).
        chunk_size   : Maximum characters per chunk (default 900).
        chunk_overlap: Characters of overlap prepended from previous chunk.

    Returns:
        List of chunk dicts compatible with the FAISS/BM25 index schema.
    """
    print(f"📖 [recursive] Loading: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Minimal normalisation
    text = text.replace('\u200d', '').replace('\u200c', '')
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Split
    separators = ['\n\n', '\n', '।', ' ', '']
    raw_chunks = _split_recursive(text, separators, chunk_size)
    raw_chunks = _add_overlap(raw_chunks, chunk_overlap)
    raw_chunks = [c for c in raw_chunks if c.strip()]

    # Build structured chunks
    chunks: List[Dict] = []
    chunk_id = 1
    current_bhag = ''
    current_parichhed = ''

    for raw in raw_chunks:
        # Update section headers found in this chunk
        for m in _BHAG_HEADER.finditer(raw):
            current_bhag = f"भाग–{_normalize_numbers(m.group(1))}"
            if m.group(2).strip():
                current_bhag += f" {m.group(2).strip()}"
        for m in _PARICHHED_HDR.finditer(raw):
            current_parichhed = f"परिच्छेद–{_normalize_numbers(m.group(1))}"
            if m.group(2).strip():
                current_parichhed += f" {m.group(2).strip()}"

        dafa_labels = _extract_dafas(raw)
        if not dafa_labels:
            dafa_str = 'Unknown'
        elif len(dafa_labels) == 1:
            dafa_str = dafa_labels[0]
        else:
            dafa_str = ', '.join(dafa_labels)

        chunks.append({
            'chunk_id':    chunk_id,
            'dafa':        dafa_str,
            'dafa_list':   dafa_labels if dafa_labels else [dafa_str],
            'text':        raw.strip(),
            'char_count':  len(raw.strip()),
            'is_merged':   len(dafa_labels) > 1,
            'has_overlap': chunk_id > 1,
            'incomplete':  False,
            'source':      source,
            'act_name':    act_name,
            'bhag':        current_bhag,
            'parichhed':   current_parichhed,
        })
        chunk_id += 1

    _print_stats("recursive", chunks)
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
