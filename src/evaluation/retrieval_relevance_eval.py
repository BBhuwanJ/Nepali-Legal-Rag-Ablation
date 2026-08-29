#!/usr/bin/env python
"""
Retrieval Relevance Evaluation (LangSmith Metric)
==================================================
Evaluates whether the retrieved context chunks are relevant to the question.

Uses Gemini as the LLM judge to assess retrieval quality on a scale of 0-1.
This metric evaluates the retriever component of the RAG pipeline.

Features:
- Incremental saving every N items
- Resume capability from partial results
- Graceful handling of API quota errors
- Gemini API key rotation for handling quota limits
- Score clamping to [0.0, 1.0]
- Temperature = 0.0 for reproducible, deterministic judgements

Precision@K metric:
  "average_precision" here is the mean of (relevant_chunks / total_chunks)
  per question — i.e., the fraction of retrieved chunks that were deemed
  relevant by the judge. This is NOT the classic Information Retrieval
  Average Precision (which sums precision-at-k over relevant documents).
  The output key is named "avg_relevant_chunk_ratio" to avoid confusion.
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
class RetrievalRelevanceResult:
    """Result of a retrieval relevance evaluation"""
    question_id: str
    score: float  # 0-1 scale (discrete: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    reasoning: str
    question: str
    num_chunks: int
    relevant_chunk_indices: List[int]  # 0-indexed positions of relevant chunks
    required_dafa_found: bool  # Whether the required dafas were found


RETRIEVAL_RELEVANCE_PROMPT = """You are an expert evaluator for a Nepali legal RAG (Retrieval-Augmented Generation) system.

Your task is to evaluate the RETRIEVAL RELEVANCE - whether the retrieved context chunks are relevant to answering the question.

## What is Retrieval Relevance?
This metric evaluates the quality of the retrieval system, NOT the final answer.
- Do the retrieved chunks contain information needed to answer the question?
- Are the chunks from the appropriate legal sections?
- Is there noise (irrelevant chunks) in the retrieval?

## Important Rules
- Evaluate ONLY the retrieved chunks, not the final answer.
- Only judge relevance based on the text explicitly present in the retrieved chunks. Do NOT assume information that is not written in the chunks.
- Each chunk is a retrieved text segment from Nepali legal documents such as the Civil Code, Criminal Code, Constitution, or related Acts. Chunks may contain section numbers (दफा), explanations, or subclauses.
- If multiple legal sections are required to answer the question, high-quality retrieval should include all key sections.
- A small number of extra but harmless chunks should not significantly reduce the score.
- Use ONLY the following scores: 1.0, 0.8, 0.6, 0.4, 0.2, 0.0.

## Evaluation Criteria:
1. **Contains Answer Information**: Do any chunks contain facts needed to answer the question?
2. **Section Accuracy**: For section lookup questions, is the correct दफा (section) retrieved?
3. **Minimal Noise**: Are most chunks relevant, or are there many irrelevant ones? Noise includes chunks from unrelated legal topics, wrong laws, or unrelated दफा numbers.
4. **Completeness**: For complex questions, are all necessary sections retrieved?

## Scoring Guidelines (use ONLY these exact values):
- **1.0**: Perfect retrieval. All chunks highly relevant, correct sections retrieved.
- **0.8**: Excellent. Most chunks relevant, key information present.
- **0.6**: Good. Some relevant chunks, but also some noise or missing info.
- **0.4**: Fair. Partially relevant, significant noise or missing key sections.
- **0.2**: Poor. Mostly irrelevant chunks, hard to answer from these.
- **0.0**: Failed. No relevant chunks retrieved, completely off-topic.

## Special Considerations:
- If the question asks about a specific दफा (section), that exact section MUST be retrieved for a score ≥ 0.8.
- For out-of-scope questions (e.g., non-Nepali law), not retrieving any legal content is actually correct — assign a high score (0.8–1.0).
- Quality > Quantity: A few highly relevant chunks are better than many weakly relevant ones.

## Input:
**Question**: {question}

**Required Dafas (Reference)**: {required_dafas}

**Retrieved Chunks**:
{chunks}

## Output Format:
Respond with a JSON object containing:
- "score": A float — MUST be exactly one of: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
- "reasoning": A brief explanation (2-3 sentences) of your score in English
- "relevant_chunk_indices": List of 0-indexed chunk numbers that are relevant (e.g., [0, 2] if chunks 1 and 3 are relevant)
- "required_dafa_found": Boolean - were the required dafas found in retrieved chunks?

Example:
{{"score": 0.8, "reasoning": "Retrieved chunks include Dafa 70 which contains marriage age requirements. Chunks 0 and 2 are directly relevant. One chunk about divorce is less relevant but not harmful.", "relevant_chunk_indices": [0, 2], "required_dafa_found": true}}

Evaluate now:"""


class RetrievalRelevanceEvaluator:
    """Evaluates retrieval relevance using Gemini as judge"""

    def __init__(self, model: str = "gemini-3.5-flash"):
        """
        Initialize the evaluator.

        Args:
            model: Gemini model to use (default: gemini-3.5-flash)
        """
        self.key_rotator = GeminiKeyRotator()
        self.model = self.key_rotator.get_model(model)
        self.model_name = model
        print(f"✅ RetrievalRelevanceEvaluator initialized with {model}")

        # State used for graceful shutdown / resume
        self._current_results: List = []
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
            if self._current_results and self._output_file:
                self._save_partial_results(self._current_results, self._current_eval_data, self._output_file)
                print(f"   💾 Partial results saved to {Path(__file__).parent / self._output_file}")
        except Exception as e:
            print(f"   ❌ Failed to save partial results: {e}")
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

    def _format_chunks(self, chunks: List[Dict], max_chars: int = 8000) -> str:
        """Format retrieved chunks for evaluation prompt"""
        if not chunks:
            return "(No chunks retrieved)"

        chunk_parts = []
        total_chars = 0

        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")[:1000]  # Truncate long chunks
            dafa = chunk.get("dafa", "N/A")
            score = chunk.get("score", 0)
            metadata = chunk.get("metadata", {})

            act_name = metadata.get("act_name", "")
            bhag = metadata.get("bhag", "")
            parichhed = metadata.get("parichhed", "")

            location = " | ".join(filter(None, [act_name, bhag, parichhed, f"दफा {dafa}"]))

            chunk_str = f"[Chunk {i} | {location} | Retrieval Score: {score:.3f}]\n{text}"

            if total_chars + len(chunk_str) > max_chars:
                chunk_parts.append(f"... ({len(chunks) - i} more chunks truncated)")
                break

            chunk_parts.append(chunk_str)
            total_chars += len(chunk_str)

        return "\n\n".join(chunk_parts)

    def evaluate_single(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        required_dafas: List[str],
        question_id: str = "unknown"
    ) -> RetrievalRelevanceResult:
        """
        Evaluate retrieval relevance for a single question.

        Args:
            question: The original question
            retrieved_chunks: List of retrieved context chunks
            required_dafas: List of expected dafa numbers for this question
            question_id: Identifier for the question

        Returns:
            RetrievalRelevanceResult with score and reasoning
        """
        chunks_text = self._format_chunks(retrieved_chunks)
        required_dafas_str = ", ".join(required_dafas) if required_dafas else "(not specified)"

        prompt = RETRIEVAL_RELEVANCE_PROMPT.format(
            question=question,
            required_dafas=required_dafas_str,
            chunks=chunks_text
        )

        try:
            result_text = self._make_api_call(prompt)

            # Normalize fenced code blocks
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

            # Validate and sanitize relevant_chunk_indices
            raw_indices = result.get("relevant_chunk_indices", [])
            valid_indices = [
                idx for idx in raw_indices
                if isinstance(idx, int) and 0 <= idx < len(retrieved_chunks)
            ]

            return RetrievalRelevanceResult(
                question_id=question_id,
                score=_snap_score(result.get("score", 0)),
                reasoning=result.get("reasoning", "No reasoning provided"),
                question=question,
                num_chunks=len(retrieved_chunks),
                relevant_chunk_indices=valid_indices,
                required_dafa_found=bool(result.get("required_dafa_found", False))
            )

        except json.JSONDecodeError as e:
            print(f"   ⚠️ JSON parse error for {question_id}: {e}")
            # Truncated-JSON fallback: salvage score via regex
            score_match = re.search(r'"score"\s*:\s*([\d.]+)', result_text)
            reasoning_match = re.search(r'"reasoning"\s*:\s*"(.*?)(?:"|$)', result_text, re.DOTALL)
            dafa_found_match = re.search(r'"required_dafa_found"\s*:\s*(true|false)', result_text, re.IGNORECASE)
            if score_match:
                salvaged_score = _snap_score(float(score_match.group(1)))
                salvaged_reasoning = reasoning_match.group(1).strip() if reasoning_match else "(reasoning truncated)"
                salvaged_dafa = dafa_found_match.group(1).lower() == "true" if dafa_found_match else False
                print(f"   🔧 Salvaged score={salvaged_score} from truncated response")
                return RetrievalRelevanceResult(
                    question_id=question_id,
                    score=salvaged_score,
                    reasoning=f"[truncated response] {salvaged_reasoning}",
                    question=question,
                    num_chunks=len(retrieved_chunks),
                    relevant_chunk_indices=[],
                    required_dafa_found=salvaged_dafa
                )
            return RetrievalRelevanceResult(
                question_id=question_id,
                score=0.0,
                reasoning=f"Failed to parse LLM response: {str(e)}",
                question=question,
                num_chunks=len(retrieved_chunks),
                relevant_chunk_indices=[],
                required_dafa_found=False
            )
        except Exception as e:
            print(f"   ❌ Error evaluating {question_id}: {e}")
            return RetrievalRelevanceResult(
                question_id=question_id,
                score=0.0,
                reasoning=f"Evaluation error: {str(e)}",
                question=question,
                num_chunks=len(retrieved_chunks),
                relevant_chunk_indices=[],
                required_dafa_found=False
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

        # avg_relevant_chunk_ratio = mean of (relevant_chunks / total_chunks) per question
        # This is the fraction of retrieved chunks deemed relevant by the judge.
        relevant_ratios = [
            len(r.relevant_chunk_indices) / r.num_chunks
            for r in results if r.num_chunks > 0
        ]
        avg_relevant_chunk_ratio = sum(relevant_ratios) / len(relevant_ratios) if relevant_ratios else 0.0

        # required_dafa_hit_rate = fraction of questions where required dafas were found
        dafa_hits = sum(1 for r in results if r.required_dafa_found)
        required_dafa_hit_rate = dafa_hits / num_evaluated if num_evaluated > 0 else 0.0

        poor_retrievals = [r for r in results if r.score < 0.5]

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
            "metric": "retrieval_relevance",
            "model_judge": self.model_name,
            "total_evaluated": num_evaluated,
            "average_score": round(avg_score, 4),
            "avg_relevant_chunk_ratio": round(avg_relevant_chunk_ratio, 4),
            "required_dafa_hit_rate": round(required_dafa_hit_rate, 4),
            "partial": True,
            "poor_retrieval_count": len(poor_retrievals),
            "by_category": {
                cat: {"count": data["count"], "avg_score": round(data["avg_score"], 4)}
                for cat, data in by_category.items()
            },
            "detailed_results": [
                {
                    "id": r.question_id,
                    "score": r.score,
                    "reasoning": r.reasoning,
                    "num_chunks": r.num_chunks,
                    "relevant_chunks": len(r.relevant_chunk_indices),
                    "relevant_chunk_indices": r.relevant_chunk_indices,
                    "required_dafa_found": r.required_dafa_found
                }
                for r in results
            ],
            "poor_retrievals": [
                {"id": r.question_id, "score": r.score, "question": r.question[:100]}
                for r in poor_retrievals
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
                        if "Evaluation error:" in reasoning or "Failed to parse LLM response:" in reasoning:
                            continue
                        evaluated_ids.add(r["id"])
                        # Restore exact saved indices rather than fabricating them
                        saved_indices = r.get("relevant_chunk_indices", [])
                        if not saved_indices:
                            # Legacy fallback: old format only stored count, not the actual indices
                            saved_indices = list(range(r.get("relevant_chunks", 0)))
                        results.append(RetrievalRelevanceResult(
                            question_id=r["id"],
                            score=_snap_score(r["score"]),
                            reasoning=r["reasoning"],
                            question="",
                            num_chunks=r.get("num_chunks", 0),
                            relevant_chunk_indices=saved_indices,
                            required_dafa_found=r.get("required_dafa_found", False)
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
        Evaluate entire dataset for retrieval relevance.

        Args:
            eval_data: List of evaluation items
            output_file: Optional file to save detailed results
            resume: If True, resume from partial results

        Returns:
            Summary statistics and detailed results
        """
        print(f"\n🔍 Evaluating {len(eval_data)} items for Retrieval Relevance...")

        results = []
        evaluated_ids: Set[str] = set()
        # Expose live state for signal handler to save on exit
        self._current_eval_data = eval_data
        self._output_file = output_file
        if resume and output_file:
            results, evaluated_ids = self._load_existing_results(output_file)
        self._current_results = results

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
                retrieved_chunks=item.get("retrieved_chunks", []),
                required_dafas=item.get("required_dafas", []),
                question_id=question_id
            )

            results.append(result)
            new_evals_since_save += 1
            print(f"Score: {result.score:.2f} ({len(result.relevant_chunk_indices)}/{result.num_chunks} relevant | dafa_found={result.required_dafa_found})")

            if output_file and new_evals_since_save >= SAVE_INTERVAL:
                self._save_partial_results(results, eval_data, output_file)
                print(f"   💾 Progress saved ({len(results)} items)")
                new_evals_since_save = 0

        # Calculate final statistics
        num_evaluated = len(results)
        total_score = sum(r.score for r in results)
        avg_score = total_score / num_evaluated if num_evaluated > 0 else 0

        # avg_relevant_chunk_ratio: mean fraction of retrieved chunks deemed relevant
        relevant_ratios = [
            len(r.relevant_chunk_indices) / r.num_chunks
            for r in results if r.num_chunks > 0
        ]
        avg_relevant_chunk_ratio = sum(relevant_ratios) / len(relevant_ratios) if relevant_ratios else 0.0

        # required_dafa_hit_rate: fraction of questions where required dafas were retrieved
        dafa_hits = sum(1 for r in results if r.required_dafa_found)
        required_dafa_hit_rate = dafa_hits / num_evaluated if num_evaluated > 0 else 0.0

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

        poor_retrievals = [r for r in results if r.score < 0.5]

        summary = {
            "metric": "retrieval_relevance",
            "model_judge": self.model_name,
            "total_evaluated": num_evaluated,
            "average_score": round(avg_score, 4),
            "avg_relevant_chunk_ratio": round(avg_relevant_chunk_ratio, 4),
            "required_dafa_hit_rate": round(required_dafa_hit_rate, 4),
            "partial": False,
            "poor_retrieval_count": len(poor_retrievals),
            "by_category": {
                cat: {"count": data["count"], "avg_score": round(data["avg_score"], 4)}
                for cat, data in by_category.items()
            },
            "detailed_results": [
                {
                    "id": r.question_id,
                    "score": r.score,
                    "reasoning": r.reasoning,
                    "num_chunks": r.num_chunks,
                    "relevant_chunks": len(r.relevant_chunk_indices),
                    "relevant_chunk_indices": r.relevant_chunk_indices,
                    "required_dafa_found": r.required_dafa_found
                }
                for r in results
            ],
            "poor_retrievals": [
                {"id": r.question_id, "score": r.score, "question": r.question[:100]}
                for r in poor_retrievals
            ]
        }

        if output_file:
            output_path = Path(__file__).parent / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Results saved to {output_path}")

        print(f"\n{'='*50}")
        print(f"📊 RETRIEVAL RELEVANCE EVALUATION SUMMARY")
        print(f"{'='*50}")
        print(f"Total Evaluated: {num_evaluated}")
        print(f"Average Score: {avg_score:.4f}")
        print(f"Avg Relevant Chunk Ratio: {avg_relevant_chunk_ratio:.4f}")
        print(f"Required Dafa Hit Rate: {required_dafa_hit_rate:.4f}")
        print(f"Poor Retrievals (score < 0.5): {len(poor_retrievals)}")
        print(f"\nBy Category:")
        for cat, data in by_category.items():
            print(f"  {cat}: {data['avg_score']:.4f} ({data['count']} items)")

        return summary


def run_retrieval_relevance_evaluation(
    input_file: str = "evalData_populated.json",
    output_file: str = "results_retrieval_relevance.json"
):
    """
    Run retrieval relevance evaluation on populated dataset.

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
    evaluator = RetrievalRelevanceEvaluator(model="gemini-3.5-flash")
    results = evaluator.evaluate_dataset(eval_data, output_file=output_file)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Retrieval Relevance evaluation")
    parser.add_argument("--input", default="evalData_populated.json", help="Input JSON file")
    parser.add_argument("--output", default="results_retrieval_relevance.json", help="Output JSON file")

    args = parser.parse_args()

    run_retrieval_relevance_evaluation(input_file=args.input, output_file=args.output)
