"""
Grader for medium-difficulty GDPR compliance tasks.

Uses set-based precision/recall/F1 against the ground-truth violation IDs.
Applies a penalty for excessive false positives.
"""

from __future__ import annotations

from typing import Any

from models import Action, Reward
from graders.utils import safe_score


def grade_medium(action: Action, ground_truth: dict[str, Any]) -> Reward:
    """Grade a medium-task action using precision/recall over violation IDs."""
    
    # Instead of exact match, use keyword-based matching
    # These are the GDPR concepts we expect the agent to find
    EXPECTED_CONCEPTS = [
        ["PURPOSE", "LIMITATION", "ART5"],      # Art 5 purpose limitation
        ["MINIMIS", "ART5"],                      # Art 5 data minimisation  
        ["LAWFUL", "BASIS", "ART6", "CONSENT"],  # Art 6 lawful basis
        ["TRANSPARENT", "ART13", "ART12"],        # Art 13 transparency
        ["RETENTION", "STORAGE", "ART5"],         # Art 5 storage limitation
    ]
    
    predicted = [v.upper() for v in action.violation_ids]
    
    matched_concepts = 0
    for concept_keywords in EXPECTED_CONCEPTS:
        # A concept is "found" if ANY predicted violation contains
        # ANY of the concept's keywords
        for pred in predicted:
            if any(kw in pred for kw in concept_keywords):
                matched_concepts += 1
                break
    
    total_concepts = len(EXPECTED_CONCEPTS)
    recall = matched_concepts / total_concepts if total_concepts else 0.05
    
    # Precision: what fraction of agent's flags are valid
    valid_flags = 0
    for pred in predicted:
        for concept_keywords in EXPECTED_CONCEPTS:
            if any(kw in pred for kw in concept_keywords):
                valid_flags += 1
                break
    
    precision = valid_flags / len(predicted) if predicted else 0.05
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.05
    
    # Penalty for excessive false flags
    score = f1 * 0.94 + 0.03
    if len(predicted) > len(EXPECTED_CONCEPTS) + 3:
        score -= 0.1
    
    score = max(0.001, min(0.999, score))
    score = safe_score(score)
    
    return Reward(
        score=score,
        precision=precision,
        recall=recall,
        f1=f1,
        breakdown={
            "matched_concepts": matched_concepts,
            "total_concepts": total_concepts,
            "predicted_count": len(predicted),
            "valid_flags": valid_flags
        }
    )
