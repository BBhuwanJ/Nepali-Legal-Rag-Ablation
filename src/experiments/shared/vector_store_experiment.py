"""
VectorStoreExperiment
======================
A path-configurable version of VectorStoreImproved that loads from
an arbitrary index directory rather than the app config paths.

Additionally, the search() method accepts a `retrieval_mode` parameter:

  'hybrid'    — Dafa boost + BM25 + semantic, without inferred routing
  'faiss_only'  — dense/semantic only (alpha=1.0, no BM25, no dafa boost)
  'bm25_only'   — pure BM25 only (alpha=0.0, no dafa boost, no semantic)

The routed hybrid extension adds label-independent single-source constraints
and multi-source diversification.
"""
from __future__ import annotations

import sys
import os
import json
import pickle
import re
import asyncio
import functools
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import faiss
import numpy as np

# ── Resolve backend ──────────────────────────────────────────────────────────
_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _EXPERIMENTS_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.rag.vector_store_improved import (
    NepaliTextProcessor,
    HybridRetriever,
    SearchResult,
)
from shared.corpus import DOCUMENTS, infer_explicit_source_keys, require_admitted_corpus
from shared.source_router import SourceRoute, classify_source_route


def _minmax_scores(scores: Dict[int, float]) -> Dict[int, float]:
    """Normalise candidate scores without changing their rank order."""
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    span = maximum - minimum
    return {
        idx: ((score - minimum) / span if span > 1e-9 else 1.0)
        for idx, score in scores.items()
    }


def _inferred_source_for_mode(
    retrieval_mode: str, source_route: SourceRoute
) -> Optional[str]:
    """Return an inferred source only for the routed-hybrid treatment."""
    if (
        retrieval_mode == "routed_hybrid"
        and source_route.route_type == "single_source"
        and len(source_route.sources) == 1
    ):
        return source_route.sources[0]
    return None


def _explicit_source_constraint(
    explicit_sources: List[str], source_route: SourceRoute
) -> Optional[str]:
    """Return a hard explicit-source constraint only for single-source intent.

    A cross-Act question can name one Act explicitly and refer to the other by
    ``दुवै ऐन`` or a strong lexical cue.  Hard-filtering to the one named Act
    after the router has classified multi-source intent defeats diversification.
    """
    if len(explicit_sources) == 1 and not source_route.diversify:
        return explicit_sources[0]
    return None


class VectorStoreExperiment:
    """
    Experiment-friendly vector store that:
      • Loads from a custom directory (not app.config paths)
      • Supports four retrieval modes for ablation
    """

    def __init__(self, index_dir: str):
        """
        Args:
            index_dir: Directory containing faiss.index, chunks.json,
                       faiss_bm25.pkl, and faiss_manifest.json.
        """
        self.index_dir  = Path(index_dir)
        self.index_file = str(self.index_dir / 'faiss.index')
        self.chunks_file = str(self.index_dir / 'chunks.json')
        self.bm25_file  = str(self.index_dir / 'faiss_bm25.pkl')
        self.manifest_file = str(self.index_dir / 'faiss_manifest.json')

        self._model_name = "Yunika/sentence-transformer-nepali"
        self.processor   = NepaliTextProcessor()
        self.model       = None
        self.index       = None
        self.chunks: List[Dict] = []
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self._dafa_to_chunks: Dict[str, List[int]] = {}
        self._chunk_dafa_cache: Dict[int, List[str]] = {}
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._embedding_cache_maxsize = 128

    # ── Loading ──────────────────────────────────────────────────────────────
    def load(self) -> None:
        """Load FAISS index, chunks, and BM25 cache from index_dir."""
        require_admitted_corpus()
        if not os.path.exists(self.index_file):
            raise FileNotFoundError(
                f"Index not found: {self.index_file}\n"
                f"Run 01_build_all_indexes.py first."
            )

        from sentence_transformers import SentenceTransformer
        print(f"🔄 Loading model ({self._model_name})…")
        self.model = SentenceTransformer(self._model_name)

        print(f"🔄 Loading FAISS index from {self.index_dir.name}/…")
        self.index = faiss.read_index(self.index_file)

        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        bm25_cache = None
        if os.path.exists(self.bm25_file):
            try:
                with open(self.bm25_file, 'rb') as f:
                    bm25_cache = pickle.load(f)
            except Exception:
                print("⚠️  BM25 cache corrupt — rebuilding…")

        self.hybrid_retriever = HybridRetriever(self.chunks, cached_data=bm25_cache)

        # Build dafa lookup
        self._dafa_to_chunks = {}
        self._chunk_dafa_cache = {}
        for idx, chunk in enumerate(self.chunks):
            nums = self._extract_chunk_dafa_numbers_raw(chunk)
            self._chunk_dafa_cache[idx] = nums
            for num in nums:
                self._dafa_to_chunks.setdefault(num, []).append(idx)

        print(f"✅ Loaded: {self.index.ntotal} vectors, {len(self.chunks)} chunks")
        if os.path.exists(self.manifest_file):
            with open(self.manifest_file, 'r', encoding='utf-8') as f:
                m = json.load(f)
            print(f"   Built at: {m.get('built_at','?')[:19]}  model={m.get('model_name','?')}")

    # ── Helper: dafa number extraction ──────────────────────────────────────
    @staticmethod
    def _extract_chunk_dafa_numbers_raw(chunk: Dict) -> List[str]:
        numbers: List[str] = []
        dafa_list = chunk.get('dafa_list')
        if isinstance(dafa_list, list) and dafa_list:
            for label in dafa_list:
                numbers.extend(
                    NepaliTextProcessor.extract_numbers_from_dafa_label(str(label))
                )
        else:
            numbers.extend(
                NepaliTextProcessor.extract_numbers_from_dafa_label(
                    str(chunk.get('dafa', ''))
                )
            )
        seen: set = set()
        unique: List[str] = []
        for n in numbers:
            if n not in seen:
                seen.add(n)
                unique.append(n)
        return unique

    # ── Sync embedding helper ─────────────────────────────────────────────────
    def _encode(self, texts: List[str]) -> np.ndarray:
        cache_key = "||".join(texts)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        embs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=len(texts),
            show_progress_bar=False,
        ).astype('float32')
        if len(self._embedding_cache) >= self._embedding_cache_maxsize:
            self._embedding_cache.pop(next(iter(self._embedding_cache)))
        self._embedding_cache[cache_key] = embs
        return embs

    # ── Main search (sync wrapper + async) ───────────────────────────────────
    def search_sync(
        self,
        query: str,
        top_k: int = 5,
        retrieval_mode: str = 'hybrid',
    ) -> List[SearchResult]:
        """
        Synchronous search (use in experiment scripts that don't have event loops).

        Args:
            query          : User query string.
            top_k          : Number of results to return.
            retrieval_mode : 'hybrid' | 'faiss_only' | 'bm25_only' |
                             'routed_hybrid'
        """
        return asyncio.run(self.search(query, top_k, retrieval_mode))

    async def search(
        self,
        query: str,
        top_k: int = 5,
        retrieval_mode: str = 'hybrid',
    ) -> List[SearchResult]:
        """
        Async search supporting four retrieval modes.

        Modes
        ------
        hybrid     : dafa-boost + BM25 + semantic (deployed behaviour)
        faiss_only : dense/semantic only (no BM25, no dafa boost)
        bm25_only  : pure BM25 only (no semantic, no dafa boost)
        routed_hybrid: hybrid fusion plus source routing/diversification
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() first.")

        if retrieval_mode not in {'hybrid', 'faiss_only', 'bm25_only', 'routed_hybrid'}:
            raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")

        base_mode = 'hybrid' if retrieval_mode == 'routed_hybrid' else retrieval_mode
        source_route = classify_source_route(query)
        route = source_route if retrieval_mode == 'routed_hybrid' else None

        query_type    = self.processor.classify_query_type(query)
        query_dafas   = self.processor.extract_dafa_numbers(query)
        explicit_sources = infer_explicit_source_keys(query)
        explicit_source = _explicit_source_constraint(explicit_sources, source_route)
        # Inferred lexical routing is part of routed_hybrid only. Keeping it
        # out of standard hybrid makes the retrieval ablation isolate the
        # source-routing contribution. Explicitly named Acts may still be
        # filtered in every mode as an unambiguous user constraint.
        inferred_source = _inferred_source_for_mode(retrieval_mode, source_route)
        source_constraint = explicit_source or inferred_source
        query_variants = self.processor.generate_query_variants(query)
        variant_specs: List[Tuple[str, Optional[str]]] = [
            (variant, None) for variant in query_variants
        ]
        if route and route.diversify:
            # Re-run the original information need inside each source.  Adding
            # the Act's formal title to the query over-ranks preambles and
            # short-title provisions instead of the substantive provision.
            variant_specs.extend(
                (query, source)
                for source in route.sources
                if source in {registered for _, registered, _ in DOCUMENTS}
            )

        # Adaptive alpha
        if base_mode == 'faiss_only':
            alpha = 1.0
        elif base_mode == 'bm25_only':
            alpha = 0.0
        elif query_type == 'section_lookup':
            alpha = 0.3
        elif query_type == 'definition':
            alpha = 0.60
        else:
            alpha = 0.55

        filter_ctx = ('query_section' if query_type == 'section_lookup'
                      else 'query_definition' if query_type == 'definition'
                      else 'query_general')

        # ── Exact-Dafa candidates (hybrid + section_lookup only) ─────────────
        # Seed exact matches, but do not return early.  An explicitly mentioned
        # Dafa can be a cross-reference inside the provision that answers the
        # question, so semantic/BM25 candidates must still be allowed into the
        # final top-k.
        direct: List[SearchResult] = []
        if base_mode == 'hybrid' and query_type == 'section_lookup' and query_dafas:
            seen_idx: set = set()
            for mention_rank, qd in enumerate(query_dafas):
                for idx in self._dafa_to_chunks.get(qd, []):
                    if idx in seen_idx:
                        continue
                    seen_idx.add(idx)
                    chunk = self.chunks[idx]
                    if source_constraint and chunk.get('source') != source_constraint:
                        continue
                    direct.append(SearchResult(
                        chunk_id=str(chunk.get('chunk_id', idx)),
                        dafa=chunk.get('dafa', ''),
                        text=chunk.get('text', ''),
                        score=max(0.80, 1.0 - 0.05 * mention_rank),
                        semantic_score=0.0,
                        keyword_score=1.0,
                        metadata={
                            'query_type': query_type,
                            'direct_dafa_match': True,
                            'retrieval_mode': retrieval_mode,
                            'source_router': source_route.to_dict(),
                            'source_constraint': source_constraint,
                            'source':   chunk.get('source', ''),
                            'act_name': chunk.get('act_name', ''),
                            'bhag':     chunk.get('bhag', ''),
                            'parichhed': chunk.get('parichhed', ''),
                        }
                    ))

        # ── Embedding encode (batch all variants) ────────────────────────────
        loop = asyncio.get_event_loop()
        all_embs = await loop.run_in_executor(
            None,
            functools.partial(self._encode, [variant for variant, _ in variant_specs])
        )

        merged: Dict[Tuple[str, str], SearchResult] = {
            (result.metadata.get('source', ''), result.chunk_id): result
            for result in direct
        }

        for v_idx, (variant, forced_source) in enumerate(variant_specs):
            q_emb = all_embs[v_idx:v_idx + 1]
            q_norm = self.processor.normalize_numbers(variant.lower())
            q_terms = self.processor.filter_stop_words(
                q_norm.split(), context=filter_ctx
            )

            multiplier = 12 if ((route and route.diversify) or source_constraint) else 6
            search_k = min(max(top_k * multiplier, top_k), len(self.chunks))

            # Dense search
            sem_scores_raw, sem_indices = self.index.search(q_emb, search_k)
            sem_dict: Dict[int, float] = {
                int(idx): float(sc)
                for sc, idx in zip(sem_scores_raw[0], sem_indices[0])
                if idx < len(self.chunks)
            }
            sem_norm = sem_dict
            if base_mode == 'hybrid' and sem_dict:
                sem_norm = _minmax_scores(sem_dict)

            # Sparse search
            bm25_dict: Dict[int, float] = {}
            if base_mode != 'faiss_only':
                bm25_cands = self.hybrid_retriever.bm25_top_candidates(
                    q_terms, top_n=search_k
                )
                for idx, sc in bm25_cands:
                    bm25_dict[idx] = sc
                for idx in sem_dict:
                    if idx not in bm25_dict:
                        bm25_dict[idx] = self.hybrid_retriever.bm25_score(q_terms, idx)

            # Min-max normalise BM25 so dense and sparse scores have compatible
            # ranges during hybrid fusion.
            if bm25_dict:
                bm25_dict = _minmax_scores(bm25_dict)

            all_cands = set(sem_dict.keys())
            if base_mode != 'faiss_only':
                all_cands.update(bm25_dict.keys())

            for idx in all_cands:
                chunk = self.chunks[idx]
                if forced_source and chunk.get('source') != forced_source:
                    continue
                # An explicit Act name disambiguates same-numbered sections.
                # Apply a hard filter only to direct section-lookup queries;
                # broader questions may legitimately require both Acts.
                if source_constraint and chunk.get('source') != source_constraint:
                    continue
                cid   = str(chunk.get('chunk_id', idx))
                s_sc  = sem_dict.get(idx, 0.0)
                s_fuse = sem_norm.get(idx, 0.0)
                b_sc  = bm25_dict.get(idx, 0.0)

                chunk_dafa_nums = self._chunk_dafa_cache.get(
                    idx, self._extract_chunk_dafa_numbers_raw(chunk)
                )
                dafa_boost = 0.0
                if base_mode != 'faiss_only' and query_dafas and chunk_dafa_nums:
                    if any(qd in chunk_dafa_nums for qd in query_dafas):
                        dafa_boost = 1.0

                # Score fusion
                if base_mode == 'faiss_only':
                    score = s_sc
                elif base_mode == 'bm25_only':
                    score = b_sc  # pure BM25 — no dafa boost, no semantic
                elif query_type == 'section_lookup' and dafa_boost > 0:
                    score = 0.15 * s_fuse + 0.25 * b_sc + 0.60 * dafa_boost
                else:
                    score = alpha * s_fuse + (1 - alpha) * b_sc + 0.10 * dafa_boost

                result = SearchResult(
                    chunk_id=cid,
                    dafa=chunk.get('dafa', ''),
                    text=chunk.get('text', ''),
                    score=score,
                    semantic_score=s_sc,
                    keyword_score=b_sc,
                    metadata={
                        'query_type': query_type,
                        'dafa_boost': dafa_boost,
                        'retrieval_mode': retrieval_mode,
                        'source_router': source_route.to_dict(),
                        'source_constraint': source_constraint,
                        'source_conditioned_variant': forced_source,
                        'source':   chunk.get('source', ''),
                        'act_name': chunk.get('act_name', ''),
                        'bhag':     chunk.get('bhag', ''),
                        'parichhed': chunk.get('parichhed', ''),
                    }
                )
                result_key = (chunk.get('source', ''), cid)
                if result_key not in merged or result.score > merged[result_key].score:
                    merged[result_key] = result

        results = sorted(merged.values(), key=lambda r: r.score, reverse=True)
        if route and route.diversify:
            return self._diversify_by_source(results, route, top_k)
        return results[:top_k]

    @staticmethod
    def _diversify_by_source(
        results: List[SearchResult], route: SourceRoute, top_k: int
    ) -> List[SearchResult]:
        """Interleave multi-source results without consulting gold labels."""
        sources = list(route.sources)
        if top_k <= 0 or len(sources) < 2:
            return results[:max(top_k, 0)]

        queues = {
            source: [r for r in results if r.metadata.get('source') == source]
            for source in sources
        }
        # Preserve the globally best candidate at rank 1. Source balancing is a
        # coverage constraint, not permission to demote the strongest result.
        selected: List[SearchResult] = results[:1]
        selected_ids = {
            (result.metadata.get('source', ''), result.chunk_id)
            for result in selected
        }
        source_counts = {
            source: sum(r.metadata.get('source') == source for r in selected)
            for source in sources
        }
        quota = max(1, top_k // len(sources))

        while len(selected) < top_k and any(
            source_counts[source] < quota for source in sources
        ):
            eligible = []
            for source in sources:
                if source_counts[source] >= quota:
                    continue
                candidate = next(
                    (result for result in queues[source]
                     if (result.metadata.get('source', ''), result.chunk_id)
                     not in selected_ids),
                    None,
                )
                if candidate is not None:
                    eligible.append(candidate)
            if not eligible:
                break
            result = max(eligible, key=lambda item: item.score)
            selected.append(result)
            selected_ids.add((result.metadata.get('source', ''), result.chunk_id))
            source = result.metadata.get('source')
            if source in source_counts:
                source_counts[source] += 1

        for result in results:
            if len(selected) >= top_k:
                break
            result_key = (result.metadata.get('source', ''), result.chunk_id)
            if result_key not in selected_ids:
                selected.append(result)
                selected_ids.add(result_key)

        for rank, result in enumerate(selected, start=1):
            result.metadata['source_router'] = route.to_dict()
            result.metadata['diversified_rank'] = rank
        return selected
