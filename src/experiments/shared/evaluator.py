"""
Evaluator Wrapper
==================
Wraps the evaluationV2 evaluators so that experiment scripts can call
them with arbitrary input/output file paths.

Two evaluation modes are provided:

  run_all_metrics()       — COMBINED single-pass (RECOMMENDED)
                            Evaluates all 4 metrics per question with one
                            shared 5-RPM limiter and resumable output.

  run_all_metrics_sequential() — legacy sequential mode (one metric at
                            a time); kept for debugging / single-metric
                            runs.

Individual helpers (run_correctness, run_groundedness, …) are also
available for running a single metric in isolation.
"""
from __future__ import annotations

import sys
import json
import os
import datetime
from pathlib import Path
from typing import Dict, Optional

from shared.split_reporting import save_answer_split_reporting

# ── Resolve paths ─────────────────────────────────────────────────────────────
_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
_BACKEND_DIR     = _EXPERIMENTS_DIR.parent
_EVAL_V2_DIR     = _BACKEND_DIR / "evaluationV2"

# Make both backend and evaluationV2 importable
for _p in [str(_BACKEND_DIR), str(_EVAL_V2_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Model used by all evaluators (must match a real Google AI model)
JUDGE_MODEL = "gemini-3.5-flash"


def _load_populated_json(populated_path: str) -> list:
    """Load eval data, filtering out ERROR and empty answers."""
    with open(populated_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    valid_data = []
    for item in data:
        if (
            item.get("answerability") == "out_of_scope"
            or item.get("category") == "out_of_scope"
        ):
            continue
        ans = item.get("generated_answer")
        if ans and not str(ans).startswith("ERROR"):
            run_metadata = item.get("run_metadata", {})
            index_dir = run_metadata.get("index_dir")
            recorded = run_metadata.get("index_fingerprint")
            if not index_dir or not recorded:
                raise RuntimeError(
                    "Populated data has no index provenance. Regenerate it; "
                    "legacy results cannot be used after a corpus/index change."
                )
            from shared.populate_utils import (
                index_manifest_fingerprint,
                retrieval_implementation_fingerprint,
            )
            current = index_manifest_fingerprint(index_dir)
            if current != recorded:
                raise RuntimeError(
                    "Populated data does not match its current index manifest. "
                    "Regenerate answers before evaluation."
                )
            if (
                run_metadata.get("retrieval_implementation_fingerprint")
                != retrieval_implementation_fingerprint()
            ):
                raise RuntimeError(
                    "Populated data predates the current retrieval implementation. "
                    "Regenerate answers before evaluation."
                )
            valid_data.append(item)

    return valid_data


# ─────────────────────────────────────────────────────────────────────────────
# Individual metric runners  (used when only one metric is needed)
# ─────────────────────────────────────────────────────────────────────────────

def _abs_to_eval_rel(output_path: str) -> str:
    """Convert an absolute path to a path relative to the evaluationV2 dir.

    The individual evaluators write their output relative to their own
    directory (evaluationV2/).  We pass a relative path so they write
    directly to the caller-specified absolute location.
    """
    return os.path.relpath(output_path, _EVAL_V2_DIR)


def run_correctness(populated_path: str, output_path: str) -> Dict:
    from correctness_eval import CorrectnessEvaluator  # type: ignore[import]
    data     = _load_populated_json(populated_path)
    evaluator = CorrectnessEvaluator(model=JUDGE_MODEL)
    result = evaluator.evaluate_dataset(data, output_file=_abs_to_eval_rel(output_path))
    return save_answer_split_reporting(result, data, output_path)


def run_groundedness(populated_path: str, output_path: str) -> Dict:
    from groundedness_eval import GroundednessEvaluator  # type: ignore[import]
    data     = _load_populated_json(populated_path)
    evaluator = GroundednessEvaluator(model=JUDGE_MODEL)
    result = evaluator.evaluate_dataset(data, output_file=_abs_to_eval_rel(output_path))
    return save_answer_split_reporting(result, data, output_path)


def run_relevance(populated_path: str, output_path: str) -> Dict:
    from relevance_eval import RelevanceEvaluator  # type: ignore[import]
    data     = _load_populated_json(populated_path)
    evaluator = RelevanceEvaluator(model=JUDGE_MODEL)
    result = evaluator.evaluate_dataset(data, output_file=_abs_to_eval_rel(output_path))
    return save_answer_split_reporting(result, data, output_path)


def run_retrieval_relevance(populated_path: str, output_path: str) -> Dict:
    from retrieval_relevance_eval import RetrievalRelevanceEvaluator  # type: ignore[import]
    data     = _load_populated_json(populated_path)
    evaluator = RetrievalRelevanceEvaluator(model=JUDGE_MODEL)
    result = evaluator.evaluate_dataset(data, output_file=_abs_to_eval_rel(output_path))
    return save_answer_split_reporting(result, data, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Combined single-pass runner (recommended for resumable all-metric runs)
# ─────────────────────────────────────────────────────────────────────────────

def run_all_metrics(
    populated_path: str,
    out_dir: str,
    strategy_name: str,
) -> Dict:
    """
    Evaluate all 4 metrics in a single quota-safe pass.

    Calls share the same prompts as individual evaluation and are paced through
    one process-wide 5-RPM limiter. Hosted LLM outputs are not guaranteed to be
    bit-identical across separate runs.

    Args:
        populated_path : Absolute path to the populated JSON file.
        out_dir        : Directory where per-metric JSON files are written.
        strategy_name  : Label used in result filenames and the summary report.

    Returns:
        Combined report dict with keys: strategy, overall_score,
        correctness, groundedness, relevance, retrieval_relevance, …
    """
    from combined_eval import CombinedEvaluator  # type: ignore[import]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = _load_populated_json(populated_path)

    print(f"\n{'='*65}")
    print(f"  📊 Combined evaluation for [{strategy_name}]  ({len(data)} questions)")
    print(f"  Mode: 4 metrics, shared 5-RPM limiter (12s between requests)")
    print(f"  Judge model: {JUDGE_MODEL}")
    print(f"{'='*65}")

    # Redirect each evaluator's output file to out_dir/<strategy>_<metric>.json
    # by temporarily monkey-patching the OUTPUT_FILES mapping inside combined_eval.
    import combined_eval  # type: ignore[import]
    original_output_files = combined_eval.OUTPUT_FILES.copy()
    combined_eval.OUTPUT_FILES = {
        "correctness":         str(out / f"{strategy_name}_correctness.json"),
        "groundedness":        str(out / f"{strategy_name}_groundedness.json"),
        "relevance":           str(out / f"{strategy_name}_relevance.json"),
        "retrieval_relevance": str(out / f"{strategy_name}_retrieval_relevance.json"),
    }
    # Also update EVAL_DIR reference so save logic uses absolute paths
    original_eval_dir   = combined_eval.EVAL_DIR
    combined_eval.EVAL_DIR = out

    try:
        evaluator = CombinedEvaluator()
        results   = evaluator.evaluate_dataset(data)
        for metric, metric_result in results.items():
            if not isinstance(metric_result, dict):
                continue
            metric_path = out / f"{strategy_name}_{metric}.json"
            results[metric] = save_answer_split_reporting(
                metric_result, data, metric_path
            )
    finally:
        # Restore originals so standalone combined_eval.py still works
        combined_eval.OUTPUT_FILES = original_output_files
        combined_eval.EVAL_DIR     = original_eval_dir

    # Build aggregate report
    scores = [
        results.get(m, {}).get("average_score")
        for m in ["correctness", "groundedness", "relevance", "retrieval_relevance"]
    ]
    valid   = [s for s in scores if isinstance(s, (int, float))]
    overall = round(sum(valid) / len(valid), 4) if valid else None
    combined_scores = [
        results.get(m, {}).get("combined_average_score")
        for m in ["correctness", "groundedness", "relevance", "retrieval_relevance"]
    ]
    combined_valid = [s for s in combined_scores if isinstance(s, (int, float))]
    combined_overall = (
        round(sum(combined_valid) / len(combined_valid), 4)
        if combined_valid else None
    )

    report = {
        "strategy":              strategy_name,
        "report_generated":      datetime.datetime.now().isoformat(),
        "llm_judge":             JUDGE_MODEL,
        "evaluation_mode":       "combined_single_pass",
        "overall_score":         overall,
        "primary_split":         results.get("correctness", {}).get("primary_split"),
        "combined_overall_score": combined_overall,
        "correctness":           results.get("correctness",         {}).get("average_score"),
        "groundedness":          results.get("groundedness",        {}).get("average_score"),
        "relevance":             results.get("relevance",           {}).get("average_score"),
        "retrieval_relevance":   results.get("retrieval_relevance", {}).get("average_score"),
        "avg_relevant_chunk_ratio": results.get("retrieval_relevance", {}).get("avg_relevant_chunk_ratio"),
        "required_dafa_hit_rate":   results.get("retrieval_relevance", {}).get("required_dafa_hit_rate"),
        "hallucination_risk_count": results.get("groundedness",        {}).get("hallucination_risk_count"),
        "total_questions":       results.get("correctness",         {}).get("total_evaluated"),
        "combined_total_questions": results.get("correctness", {}).get("combined_total_evaluated"),
        "detailed":              results,
    }

    report_path = out / f"{strategy_name}_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📋 [{strategy_name}] Report:")
    print(f"   Overall              : {overall}")
    print(f"   Correctness          : {results.get('correctness',         {}).get('average_score')}")
    print(f"   Groundedness         : {results.get('groundedness',        {}).get('average_score')}")
    print(f"   Relevance            : {results.get('relevance',           {}).get('average_score')}")
    print(f"   Retrieval Relevance  : {results.get('retrieval_relevance', {}).get('average_score')}")
    print(f"   Dafa Hit Rate        : {results.get('retrieval_relevance', {}).get('required_dafa_hit_rate')}")
    print(f"   Hallucination Risks  : {results.get('groundedness',        {}).get('hallucination_risk_count')}")
    print(f"💾 Report saved: {report_path}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Legacy sequential runner  (kept for debugging / single-metric mode)
# ─────────────────────────────────────────────────────────────────────────────

def run_all_metrics_sequential(
    populated_path: str,
    out_dir: str,
    strategy_name: str,
) -> Dict:
    """
    Run all 4 metrics one at a time (legacy mode).

    This uses the same shared request pacing and prompts as run_all_metrics(),
    but groups work by metric instead of by question.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {}

    print(f"\n{'='*65}")
    print(f"  📊 Sequential evaluation for [{strategy_name}]  (legacy mode)")
    print(f"{'='*65}")

    for metric, run_fn in [
        ("correctness",         run_correctness),
        ("groundedness",        run_groundedness),
        ("relevance",           run_relevance),
        ("retrieval_relevance", run_retrieval_relevance),
    ]:
        out_file = str(out / f"{strategy_name}_{metric}.json")
        print(f"\n▶ {metric.upper()}…")
        try:
            results[metric] = run_fn(populated_path, out_file)
            print(f"   Score: {results[metric].get('average_score', 'N/A')}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[metric] = {"error": str(e), "average_score": None}

    scores = [results.get(m, {}).get("average_score") for m in
              ["correctness", "groundedness", "relevance", "retrieval_relevance"]]
    valid   = [s for s in scores if isinstance(s, (int, float))]
    overall = round(sum(valid) / len(valid), 4) if valid else None
    combined_scores = [
        results.get(m, {}).get("combined_average_score")
        for m in ["correctness", "groundedness", "relevance", "retrieval_relevance"]
    ]
    combined_valid = [s for s in combined_scores if isinstance(s, (int, float))]
    combined_overall = (
        round(sum(combined_valid) / len(combined_valid), 4)
        if combined_valid else None
    )

    report = {
        "strategy":              strategy_name,
        "report_generated":      datetime.datetime.now().isoformat(),
        "llm_judge":             JUDGE_MODEL,
        "evaluation_mode":       "sequential",
        "overall_score":         overall,
        "primary_split":         results.get("correctness", {}).get("primary_split"),
        "combined_overall_score": combined_overall,
        "correctness":           results.get("correctness",         {}).get("average_score"),
        "groundedness":          results.get("groundedness",        {}).get("average_score"),
        "relevance":             results.get("relevance",           {}).get("average_score"),
        "retrieval_relevance":   results.get("retrieval_relevance", {}).get("average_score"),
        "total_questions":       results.get("correctness",         {}).get("total_evaluated"),
        "combined_total_questions": results.get("correctness", {}).get("combined_total_evaluated"),
        "detailed":              results,
    }

    report_path = out / f"{strategy_name}_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📋 [{strategy_name}] Report (sequential):")
    print(f"   Overall: {overall}")
    print(f"💾 Report saved: {report_path}")
    return report
