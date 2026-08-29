"""
Retrieval Metrics
==================
Computes retrieval evaluation metrics from retrieved chunks and
ground-truth required_dafas.

Metrics reported (k=5, matching the paper):
  Hit@1      — Is the top-1 chunk a match for any required dafa?
  Hit@5      — Does any of the top-5 chunks match a required dafa?
  Precision@5 — Fraction of top-5 chunks that match a required dafa
  Recall@5   — Fraction of required dafas covered in top-5
  MRR@5      — Mean Reciprocal Rank within the top-5

"Match" definition: a chunk matches a required dafa if at least one of
its dafa_list numbers (after Devanagari→Arabic normalisation) equals
the required dafa number.
"""
from __future__ import annotations

import json
import sys
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Set, Optional

from shared.split_reporting import compute_refusal_compliance


# ── Numeral normalisation ────────────────────────────────────────────────────
_DEV_TO_ARA = {
    '०':'0','१':'1','२':'2','३':'3','४':'4',
    '५':'5','६':'6','७':'7','८':'8','९':'9',
}

def _norm(text: str) -> str:
    for d, a in _DEV_TO_ARA.items():
        text = text.replace(d, a)
    return text


def _norm_dafa_number(raw: str) -> str:
    """Normalise a dafa label while preserving amendment-letter suffixes."""
    n = _norm(raw).lower()
    identifiers = re.findall(r'\d+(?:[क-हa-z])?', n)
    return identifiers[0] if identifiers else raw.strip()


def _chunk_dafa_numbers(chunk: Dict) -> Set[str]:
    """Return normalised dafa numbers present in a chunk."""
    nums: Set[str] = set()
    dafa_list = chunk.get('dafa_list')
    if isinstance(dafa_list, list) and dafa_list:
        for label in dafa_list:
            n = _norm_dafa_number(str(label))
            if n:
                nums.add(n)
    else:
        n = _norm_dafa_number(str(chunk.get('dafa', '')))
        if n:
            nums.add(n)
    return nums


def _required_set(required_dafas: List[str]) -> Set[str]:
    """Normalise the list of required dafas to Arabic numeral strings."""
    return {_norm_dafa_number(d) for d in required_dafas if d}


# ── Per-question metrics ─────────────────────────────────────────────────────
def compute_single(
    retrieved_chunks: List[Dict],
    required_dafas: List[str],
    k: int = 5,
    required_source: Optional[str] = None,
    required_sections: Optional[List[Dict]] = None,
) -> Dict:
    """
    Compute retrieval metrics for one question.

    Args:
        retrieved_chunks : Top-k retrieved chunks (dicts with 'dafa' / 'dafa_list').
        required_dafas   : Ground-truth dafa numbers (Devanagari or Arabic).
        k                : Rank cutoff (default 5).
        required_source  : Optional source key. When supplied, a matching
                           dafa from a different Act is not a true positive.
        required_sections: Optional source-qualified targets with ``source``
                           and ``dafa`` keys. Use this for cross-Act questions.

    Returns:
        Dict with hit@1, hit@k, precision@k, recall@k, mrr@k, and
        complete@k. The complete metric is especially important for cross-Act
        questions: retrieving one of two required Acts is not a complete hit.
    """
    req = _required_set(required_dafas)
    qualified_req = {
        (str(target.get('source', '')), _norm_dafa_number(str(target.get('dafa', ''))))
        for target in (required_sections or [])
        if target.get('source') and target.get('dafa')
    }
    top_k = retrieved_chunks[:k]

    if not req and not qualified_req:
        # Out-of-scope question — no required dafas to check
        return {
            'hit@1': None, 'hit@k': None,
            'precision@k': None, 'recall@k': None, 'mrr@k': None,
            'complete@k': None,
            'skipped': True,
        }

    # Per-rank match flags
    def source_matches(chunk: Dict) -> bool:
        if not required_source:
            return True
        chunk_source = chunk.get('source')
        if not chunk_source and isinstance(chunk.get('metadata'), dict):
            chunk_source = chunk['metadata'].get('source')
        return chunk_source == required_source

    def chunk_source(chunk: Dict) -> str:
        source = chunk.get('source')
        if not source and isinstance(chunk.get('metadata'), dict):
            source = chunk['metadata'].get('source')
        return str(source or '')

    def matched_targets(chunk: Dict):
        dafas = _chunk_dafa_numbers(chunk)
        if qualified_req:
            source = chunk_source(chunk)
            return {(source, dafa) for dafa in dafas} & qualified_req
        if source_matches(chunk):
            return dafas & req
        return set()

    matches = [
        bool(matched_targets(chunk))
        for chunk in top_k
    ]

    hit1 = float(matches[0]) if matches else 0.0
    hitk = float(any(matches))

    tp   = sum(matches)
    prec = tp / len(top_k) if top_k else 0.0

    covered_dafas: Set = set()
    for chunk in top_k:
        covered_dafas.update(matched_targets(chunk))
    target_count = len(qualified_req) if qualified_req else len(req)
    recall = len(covered_dafas) / target_count if target_count else 0.0
    complete = float(target_count > 0 and len(covered_dafas) == target_count)

    mrr = 0.0
    for rank, match in enumerate(matches, start=1):
        if match:
            mrr = 1.0 / rank
            break

    return {
        'hit@1':       round(hit1, 4),
        'hit@k':       round(hitk, 4),
        'precision@k': round(prec, 4),
        'recall@k':    round(recall, 4),
        'mrr@k':       round(mrr, 4),
        'complete@k':  round(complete, 4),
        'skipped':     False,
    }


# ── Dataset-level aggregation ────────────────────────────────────────────────
def compute_dataset_metrics(
    eval_data: List[Dict],
    k: int = 5,
) -> Dict:
    """
    Compute aggregate retrieval metrics over a populated eval dataset.

    Each item must have:
      'required_dafas'  : list of ground-truth dafa numbers
      'retrieved_chunks': list of retrieved chunk dicts
    It may also have a stable 'source' key. Source-aware matching prevents
    same-numbered sections in different Acts from being counted as correct.

    Args:
        eval_data : List of populated evaluation items.
        k         : Rank cutoff (default 5).

    Returns:
        Dict with per-question results and aggregate averages.
    """
    per_q: List[Dict] = []
    metric_names = (
        'hit@1', 'hit@k', 'precision@k', 'recall@k', 'mrr@k', 'complete@k'
    )
    scores_by_scope: Dict[str, Dict[str, List[float]]] = {}
    skipped_by_scope: Counter[str] = Counter()

    def ensure_scope(scope: str) -> Dict[str, List[float]]:
        if scope not in scores_by_scope:
            scores_by_scope[scope] = {metric: [] for metric in metric_names}
        return scores_by_scope[scope]

    for item in eval_data:
        qid      = item.get('id', '?')
        category = item.get('category', '')
        split = str(item.get('split') or 'unassigned')

        # Skip out-of-scope
        if category == 'out_of_scope' or item.get('answerability') == 'out_of_scope':
            per_q.append({
                'id': qid, 'split': split, 'skipped': True,
                'reason': 'out_of_scope',
            })
            skipped_by_scope[split] += 1
            skipped_by_scope['combined'] += 1
            continue

        req_dafas  = item.get('required_dafas', [])
        req_sections = item.get('required_sections', [])
        retrieved  = item.get('retrieved_chunks', [])
        required_source = item.get('source')
        if not required_source and isinstance(item.get('metadata'), dict):
            legacy_source = item['metadata'].get('source')
            required_source = {
                'मुलुकी देवानी संहिता': 'muluki_ain',
                'घरेलु हिंसा (कसूर र सजाय) ऐन': 'domestic_violence',
            }.get(legacy_source)
        if required_source in (None, '', 'both'):
            required_source = None

        result = compute_single(
            retrieved,
            req_dafas,
            k=k,
            required_source=required_source,
            required_sections=req_sections,
        )
        result['id'] = qid
        result['split'] = split
        per_q.append(result)

        if not result.get('skipped'):
            for scope in (split, 'combined'):
                bucket = ensure_scope(scope)
                for metric in metric_names:
                    bucket[metric].append(result[metric])

    def _avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    def aggregate_scope(scope: str) -> Dict:
        scores = ensure_scope(scope)
        return {
            f'hit@1':         _avg(scores['hit@1']),
            f'hit@{k}':       _avg(scores['hit@k']),
            f'precision@{k}': _avg(scores['precision@k']),
            f'recall@{k}':    _avg(scores['recall@k']),
            f'mrr@{k}':       _avg(scores['mrr@k']),
            f'complete@{k}':  _avg(scores['complete@k']),
            'n_evaluated':    len(scores['hit@1']),
            'n_skipped':      skipped_by_scope[scope],
            'k':              k,
        }

    scopes = sorted(scope for scope in scores_by_scope if scope != 'combined')
    by_split = {scope: aggregate_scope(scope) for scope in scopes}
    by_split['combined'] = aggregate_scope('combined')
    primary_split = (
        'test' if by_split.get('test', {}).get('n_evaluated', 0) else 'combined'
    )
    aggregate = by_split[primary_split]

    return {
        'aggregation_policy': {
            'primary_split': primary_split,
            'primary_use': 'paper_main_result' if primary_split == 'test' else 'diagnostic',
            'combined_use': 'supplementary_only' if primary_split == 'test' else 'primary_fallback',
        },
        'primary_split': primary_split,
        'aggregate': aggregate,
        'aggregate_combined': by_split['combined'],
        'by_split': by_split,
        'refusal': compute_refusal_compliance(eval_data),
        'per_question': per_q,
    }


# ── Convenience: load populated file and compute ──────────────────────────────
def metrics_from_file(populated_json: str, k: int = 5) -> Dict:
    """Load a populated JSON file and return retrieval metrics."""
    with open(populated_json, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    return compute_dataset_metrics(eval_data, k=k)
