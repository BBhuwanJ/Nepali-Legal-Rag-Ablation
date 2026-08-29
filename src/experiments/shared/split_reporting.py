"""Split-aware reporting for the 350-record development/test benchmark."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


REFUSAL_PATTERNS = (
    r"उत्तर दिन सकिँदैन",
    r"उत्तर दिन असमर्थ",
    r"दायराभित्र पर्दैन",
    r"स्रोत(?:मा|हरूमा).*उल्लेख छैन",
    r"कानूनी सन्दर्भ(?:मा)?.*(?:उपलब्ध|समावेश) छैन",
    r"उपलब्ध गराइएको.*(?:उत्तर|जानकारी).*(?:छैन|दिन सकिँदैन)",
    r"जानकारी उपलब्ध छैन",
    r"cannot answer",
    r"outside (?:the )?scope",
    r"not (?:available|covered) in (?:the )?(?:provided )?(?:context|sources)",
    r"insufficient (?:context|information)",
)


def _score_summary(scores: Iterable[float]) -> Dict[str, Any]:
    values = [float(score) for score in scores]
    return {
        "n_evaluated": len(values),
        "average_score": round(sum(values) / len(values), 4) if values else None,
    }


def apply_answer_split_reporting(
    result: Dict[str, Any],
    eval_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Make held-out test the primary answer metric while retaining all 350."""
    item_by_id = {str(item.get("id")): item for item in eval_data}
    scores_by_split: dict[str, list[float]] = defaultdict(list)
    combined_scores: list[float] = []

    for row in result.get("detailed_results", []):
        item = item_by_id.get(str(row.get("id")), {})
        split = str(item.get("split") or "unassigned")
        row["split"] = split
        score = row.get("score")
        if isinstance(score, (int, float)):
            scores_by_split[split].append(float(score))
            combined_scores.append(float(score))

    by_split = {
        split: _score_summary(scores)
        for split, scores in sorted(scores_by_split.items())
    }
    by_split["combined"] = _score_summary(combined_scores)
    primary_split = "test" if by_split.get("test", {}).get("n_evaluated", 0) else "combined"
    primary = by_split[primary_split]

    result["aggregation_policy"] = {
        "primary_split": primary_split,
        "primary_use": "paper_main_result" if primary_split == "test" else "diagnostic",
        "combined_use": "supplementary_only" if primary_split == "test" else "primary_fallback",
        "out_of_scope_policy": "excluded_from_answer_metric",
    }
    result["by_split"] = by_split
    result["primary_split"] = primary_split
    result["combined_average_score"] = by_split["combined"]["average_score"]
    result["combined_total_evaluated"] = by_split["combined"]["n_evaluated"]
    result["average_score"] = primary["average_score"]
    result["total_evaluated"] = primary["n_evaluated"]
    result["dataset_split_counts"] = dict(
        Counter(str(item.get("split") or "unassigned") for item in eval_data)
    )
    return result


def save_answer_split_reporting(
    result: Dict[str, Any],
    eval_data: List[Dict[str, Any]],
    output_path: str | Path,
) -> Dict[str, Any]:
    enriched = apply_answer_split_reporting(result, eval_data)
    Path(output_path).write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return enriched


def is_refusal(answer: Any) -> bool:
    text = " ".join(str(answer or "").lower().split())
    if not text or text.startswith("error"):
        return False
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in REFUSAL_PATTERNS)


def compute_refusal_compliance(eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic refusal-compliance diagnostic for out-of-scope records."""
    rows = []
    for item in eval_data:
        if not (
            item.get("answerability") == "out_of_scope"
            or item.get("category") == "out_of_scope"
        ):
            continue
        answer = item.get("generated_answer", "")
        refused = is_refusal(answer)
        rows.append({
            "id": item.get("id"),
            "split": item.get("split", "unassigned"),
            "refused": refused,
            "generation_error": not answer or str(answer).startswith("ERROR"),
        })

    by_split: dict[str, dict[str, Any]] = {}
    for split in sorted({str(row["split"]) for row in rows}):
        selected = [row for row in rows if str(row["split"]) == split]
        refused = sum(bool(row["refused"]) for row in selected)
        by_split[split] = {
            "n": len(selected),
            "refused": refused,
            "refusal_compliance": round(refused / len(selected), 4) if selected else None,
        }
    refused = sum(bool(row["refused"]) for row in rows)
    return {
        "metric": "deterministic_refusal_compliance",
        "n": len(rows),
        "refused": refused,
        "refusal_compliance": round(refused / len(rows), 4) if rows else None,
        "by_split": by_split,
        "per_question": rows,
        "limitation": "Pattern-based compliance diagnostic; human error analysis is required.",
    }
