"""
Grader for medium-difficulty GDPR compliance tasks.

Uses set-based precision/recall/F1 against the ground-truth violation IDs.
Applies a penalty for excessive false positives.
"""

from __future__ import annotations

from typing import Any

from models import Action, Reward


def grade_medium(action: Action, ground_truth: dict[str, Any]) -> Reward:
    """Grade a medium-task action using precision/recall over violation IDs.

    Scoring:
      - score = F1 over violation IDs (naturally varies with action quality)
      - penalty: −0.1 if predicted count exceeds gold count by more than 3
      - clamped to [0.0, 1.0]
    """
    gold = set(ground_truth.get("violation_ids", []))
    predicted = set(action.violation_ids)

    true_positives = gold & predicted

    precision = len(true_positives) / len(predicted) if predicted else 0.0
    recall = len(true_positives) / len(gold) if gold else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    score = f1

    # Penalty for excessive false positives
    if len(predicted) > len(gold) + 3:
        score -= 0.1

    # Clamp
    score = max(0.0, min(1.0, score))

    return Reward(
        score=score,
        precision=precision,
        recall=recall,
        f1=f1,
        breakdown={
            "true_positives": len(true_positives),
            "false_positives": len(predicted - gold),
            "false_negatives": len(gold - predicted),
            "penalty_applied": len(predicted) > len(gold) + 3,
        },
    )
