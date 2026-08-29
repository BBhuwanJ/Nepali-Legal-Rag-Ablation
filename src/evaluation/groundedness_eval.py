#!/usr/bin/env python
"""
Groundedness Evaluation (LangSmith Metric)
==========================================
Evaluates whether the generated answer is grounded in (supported by)
the retrieved context chunks.

Uses Gemini as the LLM judge to assess groundedness on a scale of 0-1.
This metric detects hallucinations - content in the answer not supported by context.

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
# Save progress every N items (set to 1 for immediate saves when debugging)
SAVE_INTERVAL = 1  # Save progress every N items

# Add backend directory to path
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Load environment variables from backend/.env
load_dotenv(BACKEND_DIR / ".env")

# Import key rotator
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
class GroundednessResult:
    """Result of a groundedness evaluation"""
    question_id: str
    score: float  # 0-1 scale (discrete: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    reasoning: str
    question: str
    generated_answer: str
    context_summary: str  # Truncated context for logging


GROUNDEDNESS_PROMPT = """You are an expert evaluator for a Nepali legal RAG (Retrieval-Augmented Generation) system.

Your task is to evaluate the GROUNDEDNESS of a generated answer - whether it is fully supported by the provided context chunks.

## What is Groundedness?
Groundedness measures whether EVERY claim in the generated answer can be traced back to the retrieved context.
- Claims NOT found in context = HALLUCINATION (penalize heavily)
- Correct interpretation/summarization of context = GOOD
- Adding external knowledge not in context = BAD (even if factually correct)

## Evaluation Criteria:
1. **No Hallucinations**: Every fact, number, section reference must be in context
2. **Faithful Citation**: Dafa (section) numbers must match exactly with context
3. **No External Knowledge**: Answer should not add information beyond context
4. **Appropriate Refusals**: If context doesn't support an answer, refusing is correct

## Scoring Guidelines (use ONLY these exact values):
- **1.0**: Perfectly grounded. Every claim traceable to context. No hallucinations.
- **0.8**: Mostly grounded. Minor reasonable inferences that align with context.
- **0.6**: Partially grounded. Some claims supported, some not clearly traceable.
- **0.4**: Weakly grounded. Several unsupported claims or citations.
- **0.2**: Poorly grounded. Major hallucinations or fabricated information.
- **0.0**: Not grounded. Answer contradicts context or entirely fabricated.

## Special Cases:
- If answer says "information not available" and context truly lacks info → Score 1.0
- If the answer is an API error message or system error → Score 0.0
- Out-of-scope refusals should receive a high score if the refusal is appropriate given the context

## Input:
**Question**: {question}

**Retrieved Context Chunks**:
{context}

**Generated Answer**: {generated_answer}

## Output Format:
Respond with a JSON object containing:
- "score": A float — MUST be exactly one of: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
- "reasoning": A brief explanation (2-3 sentences) of your score in English
- "hallucinations": List of any hallucinated claims found (empty list if none)

Example:
{{"score": 0.8, "reasoning": "The answer correctly cites Dafa 70 for marriage age which is present in context. All claims are traceable.", "hallucinations": []}}

Evaluate now:"""


class GroundednessEvaluator:
    """Evaluates groundedness of RAG answers using Gemini as judge"""

    def __init__(self, model: str = "gemini-3.5-flash"):
        """
        Initialize the evaluator.

        Args:
            model: Gemini model to use (default: gemini-3.5-flash)
        """
        self.key_rotator = GeminiKeyRotator()
        self.model = self.key_rotator.get_model(model)
        self.model_name = model
        print(f"✅ GroundednessEvaluator initialized with {model}")
        # State used for graceful shutdown / resume
        self._current_results: List[GroundednessResult] = []
        self._current_eval_data: List[Dict] = []
        self._output_file: Optional[str] = None

        # Register signal handlers to save partial progress on interruption
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception:
            # Ignore if signals can't be registered in this environment
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
        # Exit with 130 to indicate SIGINT-like termination
        sys.exit(130)

    def _make_api_call(self, prompt: str) -> str:
        """Make API call with automatic key rotation on quota errors."""
        full_prompt = f"You are an expert legal evaluation assistant. You MUST respond with ONLY valid JSON, no other text.\n\n{prompt}"
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

    def _format_context(self, chunks: List[Dict], max_chars: int = 8000) -> str:
        """Format retrieved chunks into context string"""
        if not chunks:
            return "(No context retrieved)"

        context_parts = []
        total_chars = 0

        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            dafa = chunk.get("dafa", "N/A")
            score = chunk.get("score", 0)
            metadata = chunk.get("metadata", {})

            act_name = metadata.get("act_name", "")
            bhag = metadata.get("bhag", "")
            parichhed = metadata.get("parichhed", "")

            location = " | ".join(filter(None, [act_name, bhag, parichhed, f"दफा {dafa}"]))
            chunk_str = f"[Chunk {i+1} | {location} | Retrieval Score: {score:.3f}]\n{text}"

            if total_chars + len(chunk_str) > max_chars:
                context_parts.append("... (truncated for length)")
                break

            context_parts.append(chunk_str)
            total_chars += len(chunk_str)

        return "\n\n".join(context_parts)

    def evaluate_single(
        self,
        question: str,
        generated_answer: str,
        retrieved_chunks: List[Dict],
        question_id: str = "unknown"
    ) -> GroundednessResult:
        """
        Evaluate groundedness of a single answer.

        Args:
            question: The original question
            generated_answer: The RAG-generated answer
            retrieved_chunks: List of retrieved context chunks
            question_id: Identifier for the question

        Returns:
            GroundednessResult with score and reasoning
        """
        context = self._format_context(retrieved_chunks)

        prompt = GROUNDEDNESS_PROMPT.format(
            question=question,
            context=context,
            generated_answer=generated_answer
        )

        try:
            result_text = self._make_api_call(prompt)

            # Normalize fenced code blocks and extract JSON-like content
            if result_text.startswith("```"):
                parts = result_text.split("```")
                # prefer the fenced block content when present
                if len(parts) >= 3:
                    result_text = parts[1]
                else:
                    result_text = parts[1] if len(parts) > 1 else parts[0]
                if result_text.strip().lower().startswith("json"):
                    result_text = result_text.strip()[4:]

            # Robust JSON extraction and parsing helper
            def _safe_parse_json(text: str):
                # Try to locate the first JSON object or array in the text
                m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
                candidate = m.group(1) if m else text

                # First attempt: direct json.loads
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

                # Common fixes: smart quotes -> normal quotes, remove trailing commas
                fixed = candidate.replace('\u201c', '"').replace('\u201d', '"')
                fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

                # Replace lone single quotes with double quotes when it looks like a JSON dict
                if fixed.strip().startswith("{") or fixed.strip().startswith("["):
                    fixed = re.sub(r"(?<!\\)'", '"', fixed)

                try:
                    return json.loads(fixed)
                except Exception:
                    # As a last resort, try ast.literal_eval which can handle Python-style dicts
                    try:
                        return ast.literal_eval(candidate)
                    except Exception:
                        # re-raise original JSON error to be handled by caller
                        raise json.JSONDecodeError("Could not parse JSON from LLM response", text, 0)

            result = _safe_parse_json(result_text)

            return GroundednessResult(
                question_id=question_id,
                score=_snap_score(result.get("score", 0)),
                reasoning=result.get("reasoning", "No reasoning provided"),
                question=question,
                generated_answer=generated_answer,
                context_summary=context[:500] + "..." if len(context) > 500 else context
            )

        except json.JSONDecodeError as e:
            print(f"   ⚠️ JSON parse error for {question_id}: {e}")
            # Dump the raw LLM response to a debug file for inspection
            try:
                debug_path = Path(__file__).parent / f"debug_response_{question_id}.txt"
                with open(debug_path, 'w', encoding='utf-8') as df:
                    df.write(result_text if isinstance(result_text, str) else str(result_text))
                print(f"   🐞 Wrote raw LLM response to {debug_path}")
            except Exception as ex:
                print(f"   ❌ Failed to write debug response for {question_id}: {ex}")

            # --- Truncated-JSON fallback ---
            # The LLM response was cut off mid-JSON (e.g. max_output_tokens too low).
            # Try to salvage the score and partial reasoning via regex.
            score_match = re.search(r'"score"\s*:\s*([\d.]+)', result_text)
            reasoning_match = re.search(r'"reasoning"\s*:\s*"(.*?)(?:"|$)', result_text, re.DOTALL)
            if score_match:
                salvaged_score = _snap_score(float(score_match.group(1)))
                salvaged_reasoning = reasoning_match.group(1).strip() if reasoning_match else "(reasoning truncated)"
                print(f"   🔧 Salvaged score={salvaged_score} from truncated response")
                return GroundednessResult(
                    question_id=question_id,
                    score=salvaged_score,
                    reasoning=f"[truncated response] {salvaged_reasoning}",
                    question=question,
                    generated_answer=generated_answer,
                    context_summary=""
                )

            return GroundednessResult(
                question_id=question_id,
                score=0.0,
                reasoning=f"Failed to parse LLM response: {str(e)}",
                question=question,
                generated_answer=generated_answer,
                context_summary=""
            )
        except Exception as e:
            print(f"   ❌ Error evaluating {question_id}: {e}")
            return GroundednessResult(
                question_id=question_id,
                score=0.0,
                reasoning=f"Evaluation error: {str(e)}",
                question=question,
                generated_answer=generated_answer,
                context_summary=""
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
        hallucination_risks = [r for r in results if r.score < 0.6]

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
            "metric": "groundedness",
            "model_judge": self.model_name,
            "total_evaluated": num_evaluated,
            "average_score": round(avg_score, 4),
            "partial": True,
            "hallucination_risk_count": len(hallucination_risks),
            "by_category": {
                cat: {"count": data["count"], "avg_score": round(data["avg_score"], 4)}
                for cat, data in by_category.items()
            },
            "detailed_results": [
                {"id": r.question_id, "score": r.score, "reasoning": r.reasoning}
                for r in results
            ],
            "hallucination_risks": [
                {"id": r.question_id, "score": r.score, "reasoning": r.reasoning}
                for r in hallucination_risks
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

                if existing.get("detailed_results"):
                    for r in existing["detailed_results"]:
                        reasoning = r.get("reasoning", "")
                        # Skip failed items so they get re-evaluated on next run
                        if "Evaluation error:" in reasoning or "Failed to parse LLM response:" in reasoning:
                            continue
                        evaluated_ids.add(r["id"])
                        results.append(GroundednessResult(
                            question_id=r["id"],
                            score=_snap_score(r["score"]),
                            reasoning=r["reasoning"],
                            question="",
                            generated_answer="",
                            context_summary=""
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
        Evaluate entire dataset for groundedness.

        Args:
            eval_data: List of evaluation items
            output_file: Optional file to save detailed results
            resume: If True, resume from partial results

        Returns:
            Summary statistics and detailed results
        """
        print(f"\n🔍 Evaluating {len(eval_data)} items for Groundedness...")

        results = []
        evaluated_ids: Set[str] = set()
        if resume and output_file:
            results, evaluated_ids = self._load_existing_results(output_file)

        # Expose live state for signal handler to save on exit
        self._current_results = results
        self._current_eval_data = eval_data
        self._output_file = output_file

        new_evals_since_save = 0

        items_to_eval = [
            item for item in eval_data
            if item.get("id") not in evaluated_ids and item.get("category") != "out_of_scope"
        ]

        # Total non-out_of_scope items in full dataset
        total_in_scope = sum(1 for x in eval_data if x.get("category") != "out_of_scope")

        if evaluated_ids:
            print(f"⏭️ Skipping {len(evaluated_ids)} already evaluated items")
            print(f"📋 Remaining to evaluate: {len(items_to_eval)}")

        for i, item in enumerate(items_to_eval):
            question_id = item.get("id", f"q_{i+1}")

            print(f"[{len(results)+1}/{total_in_scope}] Evaluating {question_id}...", end=" ")

            result = self.evaluate_single(
                question=item.get("question", ""),
                generated_answer=item.get("generated_answer", ""),
                retrieved_chunks=item.get("retrieved_chunks", []),
                question_id=question_id
            )

            results.append(result)
            new_evals_since_save += 1
            print(f"Score: {result.score:.2f}")

            if output_file and new_evals_since_save >= SAVE_INTERVAL:
                self._save_partial_results(results, eval_data, output_file)
                print(f"   💾 Progress saved ({len(results)} items)")
                new_evals_since_save = 0

        # Calculate final statistics
        num_evaluated = len(results)
        total_score = sum(r.score for r in results)
        avg_score = total_score / num_evaluated if num_evaluated > 0 else 0
        hallucination_risks = [r for r in results if r.score < 0.6]

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
            "metric": "groundedness",
            "model_judge": self.model_name,
            "total_evaluated": num_evaluated,
            "average_score": round(avg_score, 4),
            "partial": False,
            "hallucination_risk_count": len(hallucination_risks),
            "by_category": {
                cat: {"count": data["count"], "avg_score": round(data["avg_score"], 4)}
                for cat, data in by_category.items()
            },
            "detailed_results": [
                {"id": r.question_id, "score": r.score, "reasoning": r.reasoning}
                for r in results
            ],
            "hallucination_risks": [
                {"id": r.question_id, "score": r.score, "reasoning": r.reasoning}
                for r in hallucination_risks
            ]
        }

        if output_file:
            output_path = Path(__file__).parent / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Results saved to {output_path}")

        print(f"\n{'='*50}")
        print(f"📊 GROUNDEDNESS EVALUATION SUMMARY")
        print(f"{'='*50}")
        print(f"Total Evaluated: {num_evaluated}")
        print(f"Average Score: {avg_score:.4f}")
        print(f"Hallucination Risks (score < 0.6): {len(hallucination_risks)}")
        print(f"\nBy Category:")
        for cat, data in by_category.items():
            print(f"  {cat}: {data['avg_score']:.4f} ({data['count']} items)")

        return summary


def run_groundedness_evaluation(
    input_file: str = "evalData_populated.json",
    output_file: str = "results_groundedness.json"
):
    """
    Run groundedness evaluation on populated dataset.

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
    evaluator = GroundednessEvaluator(model="gemini-3.5-flash")
    results = evaluator.evaluate_dataset(eval_data, output_file=output_file)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Groundedness evaluation")
    parser.add_argument("--input", default="evalData_populated.json", help="Input JSON file")
    parser.add_argument("--output", default="results_groundedness.json", help="Output JSON file")

    args = parser.parse_args()

    run_groundedness_evaluation(input_file=args.input, output_file=args.output)
