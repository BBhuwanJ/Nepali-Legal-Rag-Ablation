#!/usr/bin/env python
"""
Combined Evaluation Runner — All 4 Metrics in One Pass
=======================================================
Evaluates all 4 metrics (Correctness, Groundedness, Relevance,
Retrieval Relevance) for each question in one resumable pass. Gemini calls are
made sequentially because the configured free-tier allowance is 5 RPM.

Features:
- All 4 metrics evaluated in a single pass over the dataset
- Shared 12-second request pacing after successful evaluations
- Incremental saving of all 4 result files after each question
- Resume capability: skips questions already done in ALL 4 metrics
- Graceful Ctrl+C saves partial progress
- Immediate Gemini API key rotation on rate-limit and quota failures
"""
import json
import os
import sys
import time
import signal
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from dotenv import load_dotenv

# Add backend directory to path
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv(BACKEND_DIR / ".env")

# Import all evaluators
from correctness_eval import CorrectnessEvaluator, CorrectnessResult
from groundedness_eval import GroundednessEvaluator, GroundednessResult
from relevance_eval import RelevanceEvaluator, RelevanceResult
from retrieval_relevance_eval import RetrievalRelevanceEvaluator, RetrievalRelevanceResult

EVAL_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Output filenames
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_FILES = {
    "correctness":          "results_correctness.json",
    "groundedness":         "results_groundedness.json",
    "relevance":            "results_relevance.json",
    "retrieval_relevance":  "results_retrieval_relevance.json",
}


# ─────────────────────────────────────────────────────────────────────────────
# Result containers (mirrors the individual eval scripts)
# ─────────────────────────────────────────────────────────────────────────────

def _make_correctness_summary(results: List[CorrectnessResult], eval_data: List[Dict], partial: bool, model_name: str) -> Dict:
    """Build the same summary dict as correctness_eval.py produces."""
    n = len(results)
    avg = sum(r.score for r in results) / n if n else 0

    by_difficulty: Dict = {}
    by_category: Dict = {}
    for r in results:
        item = next((x for x in eval_data if x.get("id") == r.question_id), {})
        for key, field in [("difficulty", by_difficulty), ("category", by_category)]:
            val = item.get(key, "unknown")
            field.setdefault(val, {"scores": [], "count": 0})
            field[val]["scores"].append(r.score)
            field[val]["count"] += 1

    def _avg(d):
        return {k: {"count": v["count"], "avg_score": round(sum(v["scores"]) / len(v["scores"]), 4)}
                for k, v in d.items()}

    return {
        "metric": "correctness",
        "model_judge": model_name,
        "total_evaluated": n,
        "average_score": round(avg, 4),
        "partial": partial,
        "by_difficulty": _avg(by_difficulty),
        "by_category": _avg(by_category),
        "detailed_results": [{"id": r.question_id, "score": r.score, "reasoning": r.reasoning} for r in results],
    }


def _make_groundedness_summary(results: List[GroundednessResult], eval_data: List[Dict], partial: bool, model_name: str) -> Dict:
    n = len(results)
    avg = sum(r.score for r in results) / n if n else 0
    hallucination_risks = [r for r in results if r.score < 0.6]

    by_category: Dict = {}
    for r in results:
        item = next((x for x in eval_data if x.get("id") == r.question_id), {})
        cat = item.get("category", "unknown")
        by_category.setdefault(cat, {"scores": [], "count": 0})
        by_category[cat]["scores"].append(r.score)
        by_category[cat]["count"] += 1

    def _avg(d):
        return {k: {"count": v["count"], "avg_score": round(sum(v["scores"]) / len(v["scores"]), 4)}
                for k, v in d.items()}

    return {
        "metric": "groundedness",
        "model_judge": model_name,
        "total_evaluated": n,
        "average_score": round(avg, 4),
        "partial": partial,
        "hallucination_risk_count": len(hallucination_risks),
        "by_category": _avg(by_category),
        "detailed_results": [{"id": r.question_id, "score": r.score, "reasoning": r.reasoning} for r in results],
        "hallucination_risks": [{"id": r.question_id, "score": r.score, "reasoning": r.reasoning} for r in hallucination_risks],
    }


def _make_relevance_summary(results: List[RelevanceResult], eval_data: List[Dict], partial: bool, model_name: str) -> Dict:
    n = len(results)
    avg = sum(r.score for r in results) / n if n else 0
    low_relevance = [r for r in results if r.score < 0.6]

    by_domain: Dict = {}
    by_category: Dict = {}
    for r in results:
        item = next((x for x in eval_data if x.get("id") == r.question_id), {})
        for key, field in [("legal_domain", by_domain), ("category", by_category)]:
            val = item.get(key, "unknown")
            field.setdefault(val, {"scores": [], "count": 0})
            field[val]["scores"].append(r.score)
            field[val]["count"] += 1

    def _avg(d):
        return {k: {"count": v["count"], "avg_score": round(sum(v["scores"]) / len(v["scores"]), 4)}
                for k, v in d.items()}

    return {
        "metric": "relevance",
        "model_judge": model_name,
        "total_evaluated": n,
        "average_score": round(avg, 4),
        "partial": partial,
        "low_relevance_count": len(low_relevance),
        "by_legal_domain": _avg(by_domain),
        "by_category": _avg(by_category),
        "detailed_results": [{"id": r.question_id, "score": r.score, "reasoning": r.reasoning} for r in results],
        "low_relevance_items": [{"id": r.question_id, "score": r.score, "question": r.question[:100]} for r in low_relevance],
    }


def _make_retrieval_summary(results: List[RetrievalRelevanceResult], eval_data: List[Dict], partial: bool, model_name: str) -> Dict:
    n = len(results)
    avg = sum(r.score for r in results) / n if n else 0

    relevant_ratios = [len(r.relevant_chunk_indices) / r.num_chunks for r in results if r.num_chunks > 0]
    avg_ratio = sum(relevant_ratios) / len(relevant_ratios) if relevant_ratios else 0.0

    dafa_hits = sum(1 for r in results if r.required_dafa_found)
    dafa_hit_rate = dafa_hits / n if n else 0.0

    poor_retrievals = [r for r in results if r.score < 0.5]

    by_category: Dict = {}
    for r in results:
        item = next((x for x in eval_data if x.get("id") == r.question_id), {})
        cat = item.get("category", "unknown")
        by_category.setdefault(cat, {"scores": [], "count": 0})
        by_category[cat]["scores"].append(r.score)
        by_category[cat]["count"] += 1

    def _avg(d):
        return {k: {"count": v["count"], "avg_score": round(sum(v["scores"]) / len(v["scores"]), 4)}
                for k, v in d.items()}

    return {
        "metric": "retrieval_relevance",
        "model_judge": model_name,
        "total_evaluated": n,
        "average_score": round(avg, 4),
        "avg_relevant_chunk_ratio": round(avg_ratio, 4),
        "required_dafa_hit_rate": round(dafa_hit_rate, 4),
        "partial": partial,
        "poor_retrieval_count": len(poor_retrievals),
        "by_category": _avg(by_category),
        "detailed_results": [
            {
                "id": r.question_id, "score": r.score, "reasoning": r.reasoning,
                "num_chunks": r.num_chunks, "relevant_chunks": len(r.relevant_chunk_indices),
                "relevant_chunk_indices": r.relevant_chunk_indices,
                "required_dafa_found": r.required_dafa_found,
            }
            for r in results
        ],
        "poor_retrievals": [{"id": r.question_id, "score": r.score, "question": r.question[:100]} for r in poor_retrievals],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load previously completed IDs for a given output file
# ─────────────────────────────────────────────────────────────────────────────

def _load_done_ids(output_file: str) -> Set[str]:
    """Return the set of question IDs that have valid results in an output file."""
    path = EVAL_DIR / output_file
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        done = set()
        for r in data.get("detailed_results", []):
            reasoning = r.get("reasoning", "")
            if "Evaluation error:" in reasoning or "Failed to parse LLM response:" in reasoning:
                continue
            done.add(r["id"])
        return done
    except Exception:
        return set()


# ─────────────────────────────────────────────────────────────────────────────
# Save all 4 result files atomically
# ─────────────────────────────────────────────────────────────────────────────

def _save_all(
    c_results: List, g_results: List, r_results: List, rr_results: List,
    eval_data: List[Dict],
    c_model: str, g_model: str, rel_model: str, rr_model: str,
    partial: bool
):
    """Save all four result JSON files."""
    summaries = {
        OUTPUT_FILES["correctness"]:         _make_correctness_summary(c_results,  eval_data, partial, c_model),
        OUTPUT_FILES["groundedness"]:        _make_groundedness_summary(g_results,  eval_data, partial, g_model),
        OUTPUT_FILES["relevance"]:           _make_relevance_summary(r_results,    eval_data, partial, rel_model),
        OUTPUT_FILES["retrieval_relevance"]: _make_retrieval_summary(rr_results,   eval_data, partial, rr_model),
    }
    for fname, summary in summaries.items():
        with open(EVAL_DIR / fname, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main combined evaluator
# ─────────────────────────────────────────────────────────────────────────────

class CombinedEvaluator:
    """
    Evaluates all 4 RAG metrics in a single pass over the dataset.
    For each question, 4 API calls are made sequentially and results for all
    metrics are saved before moving to the next question.
    """

    MODEL = "gemini-3.5-flash"

    def __init__(self):
        print("🚀 Initialising all 4 evaluators...")
        # All evaluators share the singleton rotator and its global RPM limiter.
        self.correctness  = CorrectnessEvaluator(model=self.MODEL)
        self.groundedness = GroundednessEvaluator(model=self.MODEL)
        self.relevance    = RelevanceEvaluator(model=self.MODEL)
        self.retrieval    = RetrievalRelevanceEvaluator(model=self.MODEL)
        print("✅ All evaluators ready.\n")

        # Accumulated results (appended in-place, so signal handler sees live data)
        self._c_results:  List[CorrectnessResult]         = []
        self._g_results:  List[GroundednessResult]        = []
        self._r_results:  List[RelevanceResult]           = []
        self._rr_results: List[RetrievalRelevanceResult]  = []
        self._eval_data:  List[Dict] = []

        self._save_lock = threading.Lock()  # Prevent concurrent file writes

        try:
            signal.signal(signal.SIGINT,  self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception:
            pass

    def _signal_handler(self, signum, frame):
        print(f"\n⚠️  Received signal {signum}. Saving partial results for all 4 metrics...")
        try:
            with self._save_lock:
                _save_all(
                    self._c_results, self._g_results, self._r_results, self._rr_results,
                    self._eval_data,
                    self.MODEL, self.MODEL, self.MODEL, self.MODEL,
                    partial=True
                )
            print("   💾 Partial results saved for all 4 metrics.")
        except Exception as e:
            print(f"   ❌ Failed to save: {e}")
        sys.exit(130)

    # ------------------------------------------------------------------
    # Per-question evaluation (4 quota-safe API calls)
    # ------------------------------------------------------------------

    def _evaluate_one(
        self,
        item: Dict,
        question_id: str
    ) -> Tuple[CorrectnessResult, GroundednessResult, RelevanceResult, RetrievalRelevanceResult]:
        """
        Call all 4 metric evaluators sequentially for a single question.
        Returns a tuple of (correctness, groundedness, relevance, retrieval_relevance).
        """
        question         = item.get("question", "")
        ground_truth     = item.get("ground_truth", "")
        generated_answer = item.get("generated_answer", "")
        retrieved_chunks = item.get("retrieved_chunks", [])
        required_dafas   = item.get("required_dafas", [])

        return (
            self.correctness.evaluate_single(
                question, ground_truth, generated_answer, question_id
            ),
            self.groundedness.evaluate_single(
                question, generated_answer, retrieved_chunks, question_id
            ),
            self.relevance.evaluate_single(
                question, generated_answer, question_id
            ),
            self.retrieval.evaluate_single(
                question, retrieved_chunks, required_dafas, question_id
            ),
        )

    # ------------------------------------------------------------------
    # Full dataset evaluation
    # ------------------------------------------------------------------

    def evaluate_dataset(self, eval_data: List[Dict]) -> Dict:
        """
        Single-pass evaluation of all 4 metrics over the entire dataset.

        Returns a dict with keys 'correctness', 'groundedness', 'relevance',
        'retrieval_relevance', each containing the same summary structure as
        the individual evaluation scripts.
        """
        self._eval_data = eval_data

        # Determine which question IDs already have results in ALL 4 files
        done_sets = {
            metric: _load_done_ids(fname)
            for metric, fname in OUTPUT_FILES.items()
        }
        # A question is "fully done" only if it's complete in every metric file.
        # This is intentionally conservative — if any metric is missing, re-run it.
        fully_done: Set[str] = (
            done_sets["correctness"]
            & done_sets["groundedness"]
            & done_sets["relevance"]
            & done_sets["retrieval_relevance"]
        )

        # Load existing results from all 4 files so we can accumulate into them
        self._c_results  = self._load_correctness_results(done_sets["correctness"],  eval_data)
        self._g_results  = self._load_groundedness_results(done_sets["groundedness"], eval_data)
        self._r_results  = self._load_relevance_results(done_sets["relevance"],      eval_data)
        self._rr_results = self._load_retrieval_results(done_sets["retrieval_relevance"], eval_data)

        # Filter dataset
        items_to_eval = [
            item for item in eval_data
            if item.get("id") not in fully_done
            and item.get("category") != "out_of_scope"
        ]
        total_in_scope = sum(1 for x in eval_data if x.get("category") != "out_of_scope")

        print(f"📋 Total in-scope questions: {total_in_scope}")
        print(f"⏭️  Already completed (all 4 metrics): {len(fully_done)}")
        print(f"🔢 Remaining to evaluate: {len(items_to_eval)}")
        print()

        for i, item in enumerate(items_to_eval):
            question_id = item.get("id", f"q_{i+1}")
            current_num = len(self._c_results) + 1

            print(f"[{current_num}/{total_in_scope}] {question_id} — evaluating 4 paced metrics...")

            c_res, g_res, r_res, rr_res = self._evaluate_one(item, question_id)

            # Print scores on one line
            print(
                f"   ✓ Correctness={c_res.score:.2f}  "
                f"Groundedness={g_res.score:.2f}  "
                f"Relevance={r_res.score:.2f}  "
                f"RetrievalRelevance={rr_res.score:.2f}"
            )

            # Accumulate
            self._c_results.append(c_res)
            self._g_results.append(g_res)
            self._r_results.append(r_res)
            self._rr_results.append(rr_res)

            # Save all 4 files after every question (SAVE_INTERVAL=1)
            with self._save_lock:
                _save_all(
                    self._c_results, self._g_results, self._r_results, self._rr_results,
                    eval_data,
                    self.MODEL, self.MODEL, self.MODEL, self.MODEL,
                    partial=(i < len(items_to_eval) - 1)
                )

            # GeminiKeyRotator applies one shared 12-second start-to-start
            # interval after successful calls. Quota failures rotate immediately.

        # Build final summaries
        return self._build_final_summaries(eval_data)

    def _build_final_summaries(self, eval_data: List[Dict]) -> Dict:
        return {
            "correctness":         _make_correctness_summary(self._c_results,  eval_data, False, self.MODEL),
            "groundedness":        _make_groundedness_summary(self._g_results,  eval_data, False, self.MODEL),
            "relevance":           _make_relevance_summary(self._r_results,    eval_data, False, self.MODEL),
            "retrieval_relevance": _make_retrieval_summary(self._rr_results,   eval_data, False, self.MODEL),
        }

    # ------------------------------------------------------------------
    # Resume helpers — load previously saved results from each file
    # ------------------------------------------------------------------

    @staticmethod
    def _load_correctness_results(done_ids: Set[str], eval_data: List[Dict]) -> List[CorrectnessResult]:
        from correctness_eval import _snap_score
        path = EVAL_DIR / OUTPUT_FILES["correctness"]
        results = []
        if not path.exists() or not done_ids:
            return results
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in data.get("detailed_results", []):
                if r["id"] not in done_ids:
                    continue
                results.append(CorrectnessResult(
                    question_id=r["id"], score=_snap_score(r["score"]),
                    reasoning=r["reasoning"], question="", ground_truth="", generated_answer=""
                ))
        except Exception as e:
            print(f"⚠️  Could not load existing correctness results: {e}")
        return results

    @staticmethod
    def _load_groundedness_results(done_ids: Set[str], eval_data: List[Dict]) -> List[GroundednessResult]:
        from groundedness_eval import _snap_score
        path = EVAL_DIR / OUTPUT_FILES["groundedness"]
        results = []
        if not path.exists() or not done_ids:
            return results
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in data.get("detailed_results", []):
                if r["id"] not in done_ids:
                    continue
                results.append(GroundednessResult(
                    question_id=r["id"], score=_snap_score(r["score"]),
                    reasoning=r["reasoning"], question="", generated_answer="", context_summary=""
                ))
        except Exception as e:
            print(f"⚠️  Could not load existing groundedness results: {e}")
        return results

    @staticmethod
    def _load_relevance_results(done_ids: Set[str], eval_data: List[Dict]) -> List[RelevanceResult]:
        from relevance_eval import _snap_score
        path = EVAL_DIR / OUTPUT_FILES["relevance"]
        results = []
        if not path.exists() or not done_ids:
            return results
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in data.get("detailed_results", []):
                if r["id"] not in done_ids:
                    continue
                results.append(RelevanceResult(
                    question_id=r["id"], score=_snap_score(r["score"]),
                    reasoning=r["reasoning"], question="", generated_answer=""
                ))
        except Exception as e:
            print(f"⚠️  Could not load existing relevance results: {e}")
        return results

    @staticmethod
    def _load_retrieval_results(done_ids: Set[str], eval_data: List[Dict]) -> List[RetrievalRelevanceResult]:
        from retrieval_relevance_eval import _snap_score
        path = EVAL_DIR / OUTPUT_FILES["retrieval_relevance"]
        results = []
        if not path.exists() or not done_ids:
            return results
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in data.get("detailed_results", []):
                if r["id"] not in done_ids:
                    continue
                saved_indices = r.get("relevant_chunk_indices", [])
                if not saved_indices:
                    saved_indices = list(range(r.get("relevant_chunks", 0)))
                results.append(RetrievalRelevanceResult(
                    question_id=r["id"], score=_snap_score(r["score"]),
                    reasoning=r["reasoning"], question="",
                    num_chunks=r.get("num_chunks", 0),
                    relevant_chunk_indices=saved_indices,
                    required_dafa_found=r.get("required_dafa_found", False)
                ))
        except Exception as e:
            print(f"⚠️  Could not load existing retrieval relevance results: {e}")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Report + pretty printer (unchanged from run_all_evaluations.py)
# ─────────────────────────────────────────────────────────────────────────────

def generate_combined_report(results: Dict, timestamp: bool = True) -> Dict:
    output_file = "evaluation_report.json"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    scores = [
        results.get("correctness",         {}).get("average_score"),
        results.get("groundedness",        {}).get("average_score"),
        results.get("relevance",           {}).get("average_score"),
        results.get("retrieval_relevance", {}).get("average_score"),
    ]
    valid_scores = [s for s in scores if isinstance(s, (int, float))]
    overall = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None

    report = {
        "report_generated": datetime.now().isoformat(),
        "llm_judge": CombinedEvaluator.MODEL,
        "evaluation_mode": "combined_single_pass",
        "overall_score": overall if overall is not None else "N/A",
        "summary": {
            "correctness":         results.get("correctness",         {}).get("average_score", "N/A"),
            "groundedness":        results.get("groundedness",        {}).get("average_score", "N/A"),
            "relevance":           results.get("relevance",           {}).get("average_score", "N/A"),
            "retrieval_relevance": results.get("retrieval_relevance", {}).get("average_score", "N/A"),
        },
        "supplementary_metrics": {
            "avg_relevant_chunk_ratio": results.get("retrieval_relevance", {}).get("avg_relevant_chunk_ratio", "N/A"),
            "required_dafa_hit_rate":   results.get("retrieval_relevance", {}).get("required_dafa_hit_rate",   "N/A"),
            "hallucination_risk_count": results.get("groundedness",        {}).get("hallucination_risk_count", "N/A"),
        },
        "total_questions_evaluated": results.get("correctness", {}).get("total_evaluated", "N/A"),
        "detailed_results": results,
    }

    # Save timestamped + latest
    for fname in ([f"evaluation_report_{ts}.json", output_file] if timestamp else [output_file]):
        with open(EVAL_DIR / fname, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"💾 Report: evaluation_report_{ts}.json  +  evaluation_report.json")
    return report


def print_final_summary(results: Dict):
    print("\n")
    print("╔" + "═"*64 + "╗")
    print("║" + " "*20 + "FINAL EVALUATION SUMMARY" + " "*20 + "║")
    print("╠" + "═"*64 + "╣")

    metrics = [
        ("Correctness",         results.get("correctness",         {}).get("average_score", "N/A")),
        ("Groundedness",        results.get("groundedness",        {}).get("average_score", "N/A")),
        ("Relevance",           results.get("relevance",           {}).get("average_score", "N/A")),
        ("Retrieval Relevance", results.get("retrieval_relevance", {}).get("average_score", "N/A")),
    ]

    for name, score in metrics:
        if isinstance(score, (int, float)):
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            score_str = f"{score:.4f}"
        else:
            bar = "░" * 20
            score_str = str(score)
        print(f"║  {name:<22} │ {bar} │ {score_str:>8}  ║")

    print("╠" + "═"*64 + "╣")

    # Supplementary metrics
    rr = results.get("retrieval_relevance", {})
    gd = results.get("groundedness", {})
    for label, val in [
        ("Avg Chunk Relevance Ratio",  rr.get("avg_relevant_chunk_ratio")),
        ("Required Dafa Hit Rate",     rr.get("required_dafa_hit_rate")),
        ("Hallucination Risk Count",   gd.get("hallucination_risk_count")),
    ]:
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        print(f"║  {label:<22}   {'':20}   {val_str:>8}  ║")

    print("╚" + "═"*64 + "╝")

    valid_scores = [s for _, s in metrics if isinstance(s, (int, float))]
    if valid_scores:
        overall = sum(valid_scores) / len(valid_scores)
        print(f"\n📈 Overall RAG Quality Score: {overall:.4f}")
        if overall >= 0.8:
            print("   ✅ Excellent! Your RAG system is performing very well.")
        elif overall >= 0.6:
            print("   ⚠️  Good, but there's room for improvement.")
        elif overall >= 0.4:
            print("   🔶 Fair. Consider improving retrieval and generation quality.")
        else:
            print("   ❌ Needs significant improvement.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Combined single-pass evaluation with shared 5-RPM pacing."
    )
    parser.add_argument("--input", default="evalData_populated.json",
                        help="Populated eval data file (default: evalData_populated.json)")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip generating the combined evaluation_report.json")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════╗
║   🔍 KanunBot — Combined RAG Evaluation (Single Pass)            ║
║   ─────────────────────────────────────────────────────          ║
║   Metrics: Correctness · Groundedness · Relevance                ║
║            Retrieval Relevance                                    ║
║   Mode: 4 metrics with shared 5-RPM request pacing               ║
║   Next request waits after each successful request              ║
╚══════════════════════════════════════════════════════════════════╝
""")

    if not os.getenv("GEMINI_API_KEY"):
        print("❌ ERROR: GEMINI_API_KEY not set in backend/.env")
        sys.exit(1)

    input_path = EVAL_DIR / args.input
    if not input_path.exists():
        print(f"❌ ERROR: {input_path} not found. Run populate_eval_data.py first.")
        sys.exit(1)

    print(f"📂 Loading data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    print(f"✅ Loaded {len(eval_data)} items\n")

    start_time = time.time()
    evaluator = CombinedEvaluator()
    results = evaluator.evaluate_dataset(eval_data)

    if not args.no_report:
        print("\n" + "="*60)
        print("📋 Generating Combined Report")
        print("="*60)
        generate_combined_report(results)

    elapsed = time.time() - start_time
    print_final_summary(results)
    print(f"\n⏱️  Total evaluation time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
