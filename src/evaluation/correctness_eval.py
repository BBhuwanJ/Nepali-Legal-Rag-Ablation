#!/usr/bin/env python
"""
Correctness Evaluation (LangSmith Metric)
==========================================
Evaluates whether the generated answer is factually correct
compared to the ground truth answer.

Uses Gemini as the LLM judge to assess correctness on a scale of 0-1.

Features:
- Incremental saving every N items
- Resume capability from partial results
- Graceful handling of API quota errors
- Gemini API key rotation for handling quota limits
- Score clamping to [0.0, 1.0]
- Temperature = 0.0 for reproducible, deterministic judgements
"""
import json
import re
import ast
import os
import sys
import signal
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai

# Configuration
SAVE_INTERVAL = 1  # Save progress every N items

# Add backend directory to path
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Add evaluationV2 to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from backend/.env
load_dotenv(BACKEND_DIR / ".env")

# Import Gemini key rotator
from gemini_key_rotator import GeminiKeyRotator

# Allowed discrete score values (for IEEE reproducibility)
ALLOWED_SCORES = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}


def _snap_score(raw: float) -> float:
    """
    Snap a raw score to the nearest allowed discrete value in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}.
    Also clamps to [0.0, 1.0] to guard against out-of-range LLM outputs.
    """
    clamped = max(0.0, min(1.0, float(raw)))
    return min(ALLOWED_SCORES, key=lambda s: abs(s - clamped))


@dataclass
class CorrectnessResult:
    """Result of a correctness evaluation"""
    question_id: str
    score: float  # 0-1 scale (discrete: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    reasoning: str
    question: str
    ground_truth: str
    generated_answer: str


CORRECTNESS_PROMPT = """You are an expert evaluator for a Nepali legal RAG (Retrieval-Augmented Generation) system.

Your task is to evaluate the CORRECTNESS of a generated answer compared to the reference ground truth answer.

## Evaluation Criteria:
- **Factual Accuracy**: Does the generated answer contain the same key facts as the ground truth?
- **Legal Precision**: Are legal terms, section numbers (दफा), and conditions correctly stated?
- **Completeness**: Does the answer cover all important points from the ground truth?
- **No Contradictions**: Does the answer avoid stating anything that contradicts the ground truth?

## Scoring Guidelines (use ONLY these exact values):
- **1.0**: Perfect or near-perfect match. All key facts correct, no contradictions.
- **0.8**: Mostly correct. Minor omissions but no factual errors.
- **0.6**: Partially correct. Some key facts present, some missing, no major errors.
- **0.4**: Weak. Missing significant information or minor factual errors.
- **0.2**: Poor. Major factual errors or mostly irrelevant content.
- **0.0**: Incorrect. Completely wrong, contradictory, or empty/error response.

## Input:
**Question**: {question}

**Ground Truth Answer**: {ground_truth}

**Generated Answer**: {generated_answer}

## Output Format:
Respond with a JSON object containing:
- "score": A float — MUST be exactly one of: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
- "reasoning": A brief explanation (2-3 sentences) of your score in English

Example:
{{"score": 0.8, "reasoning": "The answer correctly identifies the minimum marriage age as 20 years and cites Dafa 70, matching the ground truth. Minor formatting differences but factually accurate."}}

Evaluate now:"""


class CorrectnessEvaluator:
    """Evaluates correctness of RAG answers using Gemini as judge"""

    def __init__(self, model: str = "gemini-3.5-flash"):
        """
        Initialize the evaluator.

        Args:
            model: Gemini model to use (default: gemini-3.5-flash)
        """
        self.key_rotator = GeminiKeyRotator()
        self.model_name = model
        self.model = self.key_rotator.get_model(model)
        print(f"✅ CorrectnessEvaluator initialized with {model}")
        # State used for graceful shutdown / resume
        self._current_results: List[CorrectnessResult] = []
        self._current_eval_data: List[Dict] = []
        self._output_file: Optional[str] = None

        # Register signal handlers to save partial progress on interruption
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception:
            pass

    def _signal_handler(self, signum, frame):
        """Handle termination signals by saving partial results."""
        print(f"\n⚠️ Received signal {signum}, attempting to save partial results...")
        try:
            if getattr(self, '_current_results', None) and self._output_file:
                self._save_partial_results(self._current_results, self._current_eval_data, self._output_file)
                print(f"   💾 Partial results saved to {Path(__file__).parent / self._output_file}")
        except Exception as e:
            print(f"   ❌ Failed to save partial results: {e}")
        sys.exit(130)

    def _make_api_call(self, prompt: str) -> str:
        """Make API call with automatic key rotation on quota errors."""
        # Wrap prompt with JSON instruction
        full_prompt = f"""You are an expert legal evaluation assistant. You MUST respond with ONLY valid JSON, no other text.

{prompt}"""
        response = self.key_rotator.generate_content(
            full_prompt,
            model_name=self.model_name,
            request_options={"timeout": 60},
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=8192,
                response_mime_type="application/json",
            ),
        )
        return response.text.strip()

    def evaluate_single(
        self,
        question: str,
        ground_truth: str,
        generated_answer: str,
        question_id: str = "unknown"
    ) -> CorrectnessResult:
        """
        Evaluate a single question-answer pair.

        Args:
            question: The original question
            ground_truth: The reference correct answer
            generated_answer: The RAG-generated answer
            question_id: Identifier for the question

        Returns:
            CorrectnessResult with score and reasoning
        """
        prompt = CORRECTNESS_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            generated_answer=generated_answer
        )

        try:
            result_text = self._make_api_call(prompt)

            # Normalize fenced code blocks and extract JSON-like content
            if result_text.startswith("```"):
                parts = result_text.split("```")
                if len(parts) >= 3:
                    result_text = parts[1]
                else:
                    result_text = parts[1] if len(parts) > 1 else parts[0]
                if result_text.strip().lower().startswith("json"):
                    result_text = result_text.strip()[4:]

            # Robust JSON extraction and parsing helper
            def _safe_parse_json(text: str):
                m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
                candidate = m.group(1) if m else text
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
                fixed = candidate.replace('\u201c', '"').replace('\u201d', '"')
                fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
                if fixed.strip().startswith("{") or fixed.strip().startswith("["):
                    fixed = re.sub(r"(?<!\\)'", '"', fixed)
                try:
                    return json.loads(fixed)
                except Exception:
                    try:
                        return ast.literal_eval(candidate)
                    except Exception:
                        raise json.JSONDecodeError("Could not parse JSON from LLM response", text, 0)

            result = _safe_parse_json(result_text)

            return CorrectnessResult(
                question_id=question_id,
                score=_snap_score(result.get("score", 0)),
                reasoning=result.get("reasoning", "No reasoning provided"),
                question=question,
                ground_truth=ground_truth,
                generated_answer=generated_answer
            )

        except json.JSONDecodeError as e:
            print(f"   ⚠️ JSON parse error for {question_id}: {e}")
            # Truncated-JSON fallback: salvage score via regex
            score_match = re.search(r'"score"\s*:\s*([\d.]+)', result_text)
            reasoning_match = re.search(r'"reasoning"\s*:\s*"(.*?)(?:"|$)', result_text, re.DOTALL)
            if score_match:
                salvaged_score = _snap_score(float(score_match.group(1)))
                salvaged_reasoning = reasoning_match.group(1).strip() if reasoning_match else "(reasoning truncated)"
                print(f"   🔧 Salvaged score={salvaged_score} from truncated response")
                return CorrectnessResult(
                    question_id=question_id,
                    score=salvaged_score,
                    reasoning=f"[truncated response] {salvaged_reasoning}",
                    question=question,
                    ground_truth=ground_truth,
                    generated_answer=generated_answer
                )
            return CorrectnessResult(
                question_id=question_id,
                score=0.0,
                reasoning=f"Failed to parse LLM response: {str(e)}",
                question=question,
                ground_truth=ground_truth,
                generated_answer=generated_answer
            )
        except Exception as e:
            print(f"   ❌ Error evaluating {question_id}: {e}")
            return CorrectnessResult(
                question_id=question_id,
                score=0.0,
                reasoning=f"Evaluation error: {str(e)}",
                question=question,
                ground_truth=ground_truth,
                generated_answer=generated_answer
            )

    def _save_partial_results(
        self,
        results: List,
        eval_data: List[Dict],
        output_file: str
    ):
        """Save current results to file for resume capability."""
        num_evaluated = len(results)
        total_score = sum(r.score for r in results)
        avg_score = total_score / num_evaluated if num_evaluated > 0 else 0

        # Group by difficulty
        by_difficulty = {}
        for result in results:
            item = next((x for x in eval_data if x.get("id") == result.question_id), {})
            difficulty = item.get("difficulty", "unknown")
            if difficulty not in by_difficulty:
                by_difficulty[difficulty] = {"scores": [], "count": 0}
            by_difficulty[difficulty]["scores"].append(result.score)
            by_difficulty[difficulty]["count"] += 1

        for diff, data in by_difficulty.items():
            data["avg_score"] = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0

        # Group by category
        by_category = {}
        for result in results:
            item = next((x for x in eval_data if x.get("id") == result.question_id), {})
            category = item.get("category", "unknown")
            if category not in by_category:
                by_category[category] = {"scores": [], "count": 0}
            by_category[category]["scores"].append(result.score)
            by_category[category]["count"] += 1

        for cat, data in by_category.items():
            data["avg_score"] = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0

        summary = {
            "metric": "correctness",
            "model_judge": self.model_name,
            "total_evaluated": num_evaluated,
            "average_score": round(avg_score, 4),
            "partial": True,  # Flag to indicate partial results
            "by_difficulty": {
                diff: {"count": data["count"], "avg_score": round(data["avg_score"], 4)}
                for diff, data in by_difficulty.items()
            },
            "by_category": {
                cat: {"count": data["count"], "avg_score": round(data["avg_score"], 4)}
                for cat, data in by_category.items()
            },
            "detailed_results": [
                {
                    "id": r.question_id,
                    "score": r.score,
                    "reasoning": r.reasoning
                }
                for r in results
            ]
        }

        output_path = Path(__file__).parent / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def _load_existing_results(self, output_file: str) -> tuple[List, Set[str]]:
        """Load existing partial results for resume."""
        output_path = Path(__file__).parent / output_file
        results = []
        evaluated_ids: Set[str] = set()

        if output_path.exists():
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)

                # Check if these are valid partial results (not error-filled)
                if existing.get("detailed_results"):
                    # Filter out results that were errors (score 0 with error message)
                    for r in existing["detailed_results"]:
                        reasoning = r.get("reasoning", "")
                        # Skip failed items so they get re-evaluated on next run
                        if "Evaluation error:" in reasoning or "Failed to parse LLM response:" in reasoning:
                            continue
                        evaluated_ids.add(r["id"])
                        # Reconstruct result object
                        results.append(CorrectnessResult(
                            question_id=r["id"],
                            score=_snap_score(r["score"]),
                            reasoning=r["reasoning"],
                            question="",  # Not stored in results
                            ground_truth="",
                            generated_answer=""
                        ))

                    if evaluated_ids:
                        print(f"📂 Found {len(evaluated_ids)} valid previous results, resuming...")
            except Exception as e:
                print(f"⚠️ Could not load existing results: {e}")

        return results, evaluated_ids

    def evaluate_dataset(
        self,
        eval_data: List[Dict],
        output_file: Optional[str] = None,
        resume: bool = True
    ) -> Dict:
        """
        Evaluate entire dataset for correctness.

        Args:
            eval_data: List of evaluation items with question, ground_truth, generated_answer
            output_file: Optional file to save detailed results
            resume: If True, resume from partial results

        Returns:
            Summary statistics and detailed results
        """
        print(f"\n🔍 Evaluating {len(eval_data)} items for Correctness...")

        # Load existing results if resuming
        results = []
        evaluated_ids: Set[str] = set()
        if resume and output_file:
            results, evaluated_ids = self._load_existing_results(output_file)

        # Expose live state to signal handler BEFORE the loop begins
        self._current_results = results
        self._current_eval_data = eval_data
        self._output_file = output_file

        # Track new evaluations for incremental save
        new_evals_since_save = 0

        # Count items to evaluate (excluding already done and out_of_scope)
        items_to_eval = [
            item for item in eval_data
            if item.get("id") not in evaluated_ids and item.get("category") != "out_of_scope"
        ]
        total_remaining = len(items_to_eval)

        # Total non-out_of_scope items in full dataset
        total_in_scope = sum(1 for x in eval_data if x.get("category") != "out_of_scope")

        if evaluated_ids:
            print(f"⏭️ Skipping {len(evaluated_ids)} already evaluated items")
            print(f"📋 Remaining to evaluate: {total_remaining}")

        for i, item in enumerate(items_to_eval):
            question_id = item.get("id", f"q_{i+1}")

            current_num = len(results) + 1
            print(f"[{current_num}/{total_in_scope}] Evaluating {question_id}...", end=" ")

            result = self.evaluate_single(
                question=item.get("question", ""),
                ground_truth=item.get("ground_truth", ""),
                generated_answer=item.get("generated_answer", ""),
                question_id=question_id
            )

            results.append(result)
            new_evals_since_save += 1
            print(f"Score: {result.score:.2f}")

            # GeminiKeyRotator applies the 12-second cooldown after a success.

            # Incremental save every SAVE_INTERVAL items
            if output_file and new_evals_since_save >= SAVE_INTERVAL:
                self._save_partial_results(results, eval_data, output_file)
                print(f"   💾 Progress saved ({len(results)} items)")
                new_evals_since_save = 0

        # Calculate final statistics
        num_evaluated = len(results)
        total_score = sum(r.score for r in results)
        avg_score = total_score / num_evaluated if num_evaluated > 0 else 0

        # Group by difficulty
        by_difficulty = {}
        for result in results:
            item = next((x for x in eval_data if x.get("id") == result.question_id), {})
            difficulty = item.get("difficulty", "unknown")
            if difficulty not in by_difficulty:
                by_difficulty[difficulty] = {"scores": [], "count": 0}
            by_difficulty[difficulty]["scores"].append(result.score)
            by_difficulty[difficulty]["count"] += 1

        for diff, data in by_difficulty.items():
            data["avg_score"] = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0

        # Group by category
        by_category = {}
        for result in results:
            item = next((x for x in eval_data if x.get("id") == result.question_id), {})
            category = item.get("category", "unknown")
            if category not in by_category:
                by_category[category] = {"scores": [], "count": 0}
            by_category[category]["scores"].append(result.score)
            by_category[category]["count"] += 1

        for cat, data in by_category.items():
            data["avg_score"] = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0

        summary = {
            "metric": "correctness",
            "model_judge": self.model_name,
            "total_evaluated": num_evaluated,
            "average_score": round(avg_score, 4),
            "partial": False,  # Complete results
            "by_difficulty": {
                diff: {"count": data["count"], "avg_score": round(data["avg_score"], 4)}
                for diff, data in by_difficulty.items()
            },
            "by_category": {
                cat: {"count": data["count"], "avg_score": round(data["avg_score"], 4)}
                for cat, data in by_category.items()
            },
            "detailed_results": [
                {
                    "id": r.question_id,
                    "score": r.score,
                    "reasoning": r.reasoning
                }
                for r in results
            ]
        }

        # Save final results
        if output_file:
            output_path = Path(__file__).parent / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Results saved to {output_path}")

        # Print summary
        print(f"\n{'='*50}")
        print(f"📊 CORRECTNESS EVALUATION SUMMARY")
        print(f"{'='*50}")
        print(f"Total Evaluated: {num_evaluated}")
        print(f"Average Score: {avg_score:.4f}")
        print(f"\nBy Difficulty:")
        for diff, data in by_difficulty.items():
            print(f"  {diff}: {data['avg_score']:.4f} ({data['count']} items)")
        print(f"\nBy Category:")
        for cat, data in by_category.items():
            print(f"  {cat}: {data['avg_score']:.4f} ({data['count']} items)")

        return summary


def run_correctness_evaluation(
    input_file: str = "evalData_populated.json",
    output_file: str = "results_correctness.json"
):
    """
    Run correctness evaluation on populated dataset.

    Args:
        input_file: JSON file with populated eval data
        output_file: Output file for results
    """
    eval_dir = Path(__file__).parent
    input_path = eval_dir / input_file

    # Load evaluation data
    print(f"📂 Loading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    print(f"✅ Loaded {len(eval_data)} items")

    # Initialize evaluator and run
    evaluator = CorrectnessEvaluator(model="gemini-3.5-flash")
    results = evaluator.evaluate_dataset(eval_data, output_file=output_file)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Correctness evaluation")
    parser.add_argument("--input", default="evalData_populated.json", help="Input JSON file")
    parser.add_argument("--output", default="results_correctness.json", help="Output JSON file")

    args = parser.parse_args()

    run_correctness_evaluation(input_file=args.input, output_file=args.output)
