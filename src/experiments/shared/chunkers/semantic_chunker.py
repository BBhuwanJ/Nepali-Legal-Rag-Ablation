"""
Semantic Chunker — Chunking Strategy 3
========================================
Groups sentences/paragraphs by embedding cosine-similarity WITHOUT
enforcing dafa boundaries.  Chunks grow until either the similarity
drops below threshold or the size ceiling is hit.

Uses the same Nepali sentence-transformer model as the rest of LawBot.
"""
from __future__ import annotations

import re
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# ── Resolve project root ────────────────────────────────────────────────────
_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _EXPERIMENTS_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from sklearn.metrics.pairwise import cosine_similarity


# ── Lazy model loader ────────────────────────────────────────────────────────
_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("🔄 [semantic] Loading Nepali sentence-transformer...")
        _model = SentenceTransformer("Yunika/sentence-transformer-nepali")
        print("✅ [semantic] Model loaded")
    return _model


# ── Sentence splitter ────────────────────────────────────────────────────────
def _split_sentences(text: str) -> List[str]:
    """Split Nepali legal text without detaching numbered provisions.

    An ASCII full stop is structural in this corpus (``१२.``) and also occurs
    in abbreviations such as ``नं.``.  Treating it as a sentence terminator
    produced hundreds of meaningless chunks containing only a dafa number.
    Nepali danda and line boundaries are the reliable separators here.
    """
    parts = re.split(r'(?<=।)\s+|\r?\n+', text)
    sentences = []
    for p in parts:
        p = p.strip()
        if p:
            sentences.append(p)
    return sentences


def _split_oversized_units(units: List[str], max_chars: int) -> List[str]:
    """Split overlong units at legal-clause or word boundaries."""
    output: List[str] = []
    for unit in units:
        remaining = unit.strip()
        while len(remaining) > max_chars:
            window = remaining[: max_chars + 1]
            cut = max(
                window.rfind('।'),
                window.rfind(';'),
                window.rfind(','),
                window.rfind(' '),
            )
            # Avoid creating a pathologically small prefix when punctuation is
            # near the beginning of the window.
            if cut < max_chars // 2:
                cut = max_chars
            else:
                cut += 1
            output.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        if remaining:
            output.append(remaining)
    return output


def _coalesce_small_groups(
    groups: List[List[str]],
    min_chars: int,
    max_chars: int,
) -> List[List[str]]:
    """Attach tiny groups to an adjacent group when the ceiling permits."""
    if not groups:
        return []

    def group_len(group: List[str]) -> int:
        return len(' '.join(group))

    result: List[List[str]] = []
    pending: List[str] = []
    for group in groups:
        if pending and group_len(pending + group) <= max_chars:
            candidate = pending + group
            pending = []
        elif pending:
            result.append(pending)
            pending = []
            candidate = group
        else:
            candidate = group
        if group_len(candidate) < min_chars:
            if result and group_len(result[-1] + candidate) <= max_chars:
                result[-1].extend(candidate)
            else:
                pending = candidate
            continue
        result.append(candidate)

    if pending:
        if result and group_len(result[-1] + pending) <= max_chars:
            result[-1].extend(pending)
        else:
            result.append(pending)
    return result


# ── Semantic grouping ─────────────────────────────────────────────────────────
def _semantic_group(
    sentences: List[str],
    model,
    sim_threshold: float = 0.75,
    max_chars: int = 900,
    min_chars: int = 100,
) -> List[List[str]]:
    """
    Group sentences into semantically coherent chunks.

    A new group is started when:
      • the similarity between the current sentence and the group centroid
        drops below sim_threshold, OR
      • adding the sentence would exceed max_chars.
    """
    sentences = _split_oversized_units(sentences, max_chars=max_chars)
    if not sentences:
        return []

    embeddings = model.encode(sentences, convert_to_numpy=True,
                              normalize_embeddings=True,
                              show_progress_bar=False)

    groups: List[List[str]] = []
    current_group: List[str] = [sentences[0]]
    current_embs: List[np.ndarray] = [embeddings[0]]
    current_len = len(sentences[0])

    for i in range(1, len(sentences)):
        sent = sentences[i]
        emb  = embeddings[i]

        # Centroid of current group
        centroid = np.mean(current_embs, axis=0, keepdims=True)
        sim = float(cosine_similarity([emb], centroid)[0][0])

        would_exceed = (current_len + 1 + len(sent)) > max_chars
        similar_enough = sim >= sim_threshold

        if similar_enough and not would_exceed:
            current_group.append(sent)
            current_embs.append(emb)
            current_len += 1 + len(sent)
        else:
            groups.append(current_group)
            current_group = [sent]
            current_embs  = [emb]
            current_len   = len(sent)

    if current_group:
        groups.append(current_group)

    return _coalesce_small_groups(
        groups,
        min_chars=min_chars,
        max_chars=max_chars,
    )


# ── Metadata helpers (same as recursive_chunker) ─────────────────────────────
_DAFA_IN_TEXT = re.compile(
    r'^([०-९\d]+(?:[क-हA-Za-z])?)(?:[.)]|\s+(?=[^\d\s]))\s*',
    re.MULTILINE,
)
_BHAG_HEADER   = re.compile(r'^भाग[–\-]?\s*([०-९\d]+)\s*(.*?)$', re.MULTILINE)
_PARICHHED_HDR = re.compile(r'^परिच्छेद[–\-]?\s*([०-९\d]+)\s*(.*?)$', re.MULTILINE)
_DEV = {'०':'0','१':'1','२':'2','३':'3','४':'4','५':'5','६':'6','७':'7','८':'8','९':'9'}

def _norm_nums(t: str) -> str:
    for d, a in _DEV.items():
        t = t.replace(d, a)
    return t

def _extract_dafas(text: str) -> List[str]:
    seen: set = set()
    labels: List[str] = []
    for m in _DAFA_IN_TEXT.finditer(text):
        lbl = m.group(1)
        if lbl not in seen:
            seen.add(lbl)
            labels.append(lbl)
    return labels


# ── Public API ───────────────────────────────────────────────────────────────
def chunk_document_semantic(
    file_path: str,
    source: str = "muluki_ain",
    act_name: str = "मुलुकी देवानी (संहिता) ऐन, २०७४",
    sim_threshold: float = 0.75,
    max_chars: int = 900,
    min_chars: int = 100,
    add_overlap: bool = True,
    overlap_chars: int = 150,
) -> List[Dict]:
    """
    Chunk a Nepali legal document using semantic similarity grouping.

    Sentences are grouped by embedding cosine similarity; no dafa
    boundary enforcement is applied.

    Args:
        file_path     : Path to the source .txt document.
        source        : Source key.
        act_name      : Display name of the act (Nepali).
        sim_threshold : Cosine-similarity threshold for grouping (default 0.75).
        max_chars     : Hard ceiling on chunk size (default 900 chars).
        min_chars     : Merge tiny groups into adjacent context (default 100).
        add_overlap   : Prepend tail of previous chunk as context (default True).
        overlap_chars : Number of overlap characters (default 150).

    Returns:
        List of chunk dicts compatible with the FAISS/BM25 index schema.
    """
    print(f"📖 [semantic] Loading: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Minimal normalisation
    text = text.replace('\u200d', '').replace('\u200c', '')
    text = re.sub(r'\n{3,}', '\n\n', text)

    sentences = _split_sentences(text)
    print(f"   Sentences extracted: {len(sentences)}")

    model  = _get_model()
    groups = _semantic_group(sentences, model,
                             sim_threshold=sim_threshold,
                             max_chars=max_chars,
                             min_chars=min_chars)
    print(f"   Semantic groups formed: {len(groups)}")

    # Build structured chunks
    chunks: List[Dict] = []
    chunk_id = 1
    current_bhag = ''
    current_parichhed = ''
    raw_texts: List[str] = []

    for group in groups:
        chunk_text = ' '.join(group).strip()
        if not chunk_text:
            continue
        raw_texts.append(chunk_text)

    for i, chunk_text in enumerate(raw_texts):
        # Track section headers
        for m in _BHAG_HEADER.finditer(chunk_text):
            current_bhag = f"भाग–{_norm_nums(m.group(1))}"
            if m.group(2).strip():
                current_bhag += f" {m.group(2).strip()}"
        for m in _PARICHHED_HDR.finditer(chunk_text):
            current_parichhed = f"परिच्छेद–{_norm_nums(m.group(1))}"
            if m.group(2).strip():
                current_parichhed += f" {m.group(2).strip()}"

        # Add overlap from previous chunk
        has_overlap = False
        final_text = chunk_text
        if add_overlap and i > 0:
            tail = raw_texts[i - 1][-overlap_chars:]
            for boundary in ['।', '\n', ' ']:
                idx = tail.rfind(boundary)
                if idx > 0:
                    tail = tail[idx + 1:].strip()
                    break
            if tail:
                final_text = tail + '\n\n' + chunk_text
                has_overlap = True

        dafa_labels = _extract_dafas(final_text)
        dafa_str    = ', '.join(dafa_labels) if dafa_labels else 'Unknown'

        chunks.append({
            'chunk_id':   chunk_id,
            'dafa':       dafa_str,
            'dafa_list':  dafa_labels if dafa_labels else [dafa_str],
            'text':       final_text,
            'char_count': len(final_text),
            'is_merged':  len(dafa_labels) > 1,
            'has_overlap': has_overlap,
            'incomplete': False,
            'source':     source,
            'act_name':   act_name,
            'bhag':       current_bhag,
            'parichhed':  current_parichhed,
        })
        chunk_id += 1

    _print_stats("semantic", chunks)
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
