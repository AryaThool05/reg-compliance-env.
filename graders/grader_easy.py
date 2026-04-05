"""
Grader for easy-difficulty GDPR compliance tasks.

Scoring is based on whether the agent found any violation and whether
it correctly identified Article 6 as the relevant regulation.
Score varies based on the quality of the action — never constant.
"""

from __future__ import annotations

from typing import Any

from models import Action, Reward


def grade_easy(action: Action, ground_truth: dict[str, Any]) -> Reward:
    """Grade an easy-task action against the ground truth.

    Scoring breakdown:
      +0.5 — at least one violation identified (violation_flag)
      +0.5 — Article 6 correctly cited in any violation_id (article_cite)

    Precision / recall are computed over violation identification.
    """
    score = 0.0
    violation_flag = 0.0
    article_cite = 0.0

    # +0.5 if any violation was found
    if action.violation_ids:
        score += 0.5
        violation_flag = 0.5

    # +0.5 if "6" appears in any violation_id string
    if any("6" in vid for vid in action.violation_ids):
        score += 0.5
        article_cite = 0.5

    # Precision: perfect if exactly 1 violation reported, otherwise partial
    if action.violation_ids:
        precision = 1.0 if len(action.violation_ids) == 1 else 0.5
    else:
        precision = 0.0

    # Recall: 1.0 if at least one violation found, else 0.0
    recall = 1.0 if action.violation_ids else 0.0

    # F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    # Clamp score
    score = max(0.0, min(1.0, score))

    return Reward(
        score=score,
        precision=precision,
        recall=recall,
        f1=f1,
        breakdown={
            "violation_flag": violation_flag,
            "article_cite": article_cite,
        },
    )
