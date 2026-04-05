"""
Grader for hard-difficulty GDPR compliance tasks.

Evaluates three dimensions:
  1. Were fixed violations correctly identified?
  2. Were new violations correctly detected (or absence confirmed)?
  3. Was a substantive fix suggestion provided?
"""

from __future__ import annotations

from typing import Any

from models import Action, Reward


def _overlap(predicted: list[str], expected: list[str]) -> int:
    """Count how many expected items appear in the predicted list."""
    return len(set(predicted) & set(expected))


def grade_hard(action: Action, ground_truth: dict[str, Any]) -> Reward:
    """Grade a hard-task action across three weighted dimensions.

    Dimensions:
      dim1 (0.35) — fixed violations correctly identified
      dim2 (0.35) — new violations detected (empty list = correct absence)
      dim3 (0.30) — fix suggestion quality (length > 50 chars = full credit)

    Score is clamped to [0.0, 1.0].
    """
    fixed = ground_truth.get("fixed_violations", [])
    new = ground_truth.get("new_violations", [])

    # --- Dimension 1: fixed violations ---
    if fixed:
        score1 = _overlap(action.violation_ids, fixed) / len(fixed)
    else:
        # No fixed violations expected; reward if agent also found none matching
        score1 = 1.0

    # --- Dimension 2: new violations ---
    if new:
        score2 = _overlap(action.violation_ids, new) / len(new)
    else:
        # No new violations expected — full credit if agent didn't hallucinate new ones
        # (we can't perfectly distinguish "new" from agent output, so give full credit)
        score2 = 1.0

    # --- Dimension 3: fix suggestion quality ---
    if action.fix_suggestion and len(action.fix_suggestion) > 50:
        score3 = 1.0
    elif action.fix_suggestion:
        score3 = 0.3
    else:
        score3 = 0.0

    # Weighted final score
    final_score = 0.35 * score1 + 0.35 * score2 + 0.30 * score3
    final_score = max(0.0, min(1.0, final_score))

    # Metrics (treat all violation_ids as a single set for P/R)
    all_expected = set(fixed + new + ground_truth.get("remaining_violations", []))
    predicted = set(action.violation_ids)
    tp = len(all_expected & predicted)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(all_expected) if all_expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return Reward(
        score=final_score,
        precision=precision,
        recall=recall,
        f1=f1,
        breakdown={
            "dim1_fixed_violations": score1,
            "dim2_new_violations": score2,
            "dim3_fix_suggestion": score3,
            "weight_dim1": 0.35,
            "weight_dim2": 0.35,
            "weight_dim3": 0.30,
        },
    )
