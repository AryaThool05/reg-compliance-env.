"""
Easy task: single-clause consent violation check against GDPR Article 6.

The agent receives one clearly violating policy clause and must identify
that it lacks a valid lawful basis for data processing.
"""

from __future__ import annotations

from typing import Any

from models import Observation


class TaskEasy:
    """Single-clause GDPR compliance check (Article 6 — Lawful Basis)."""

    POLICY_CLAUSE = (
        "We share your personal data with marketing partners without "
        "requiring your explicit consent."
    )

    def get_observation(self, gdpr_cache: dict[str, Any]) -> Observation:
        """Build an easy-difficulty observation from the GDPR cache.

        Uses Article 6 text and a single obviously-violating clause.
        """
        art6 = gdpr_cache.get("6", {})
        regulation_text = art6.get("full_text", "") if isinstance(art6, dict) else str(art6)

        return Observation(
            regulation_text=regulation_text,
            policy_clause=self.POLICY_CLAUSE,
            task_id="easy",
            article_ref="Article 6",
            context={"source": "synthetic", "difficulty": "easy"},
        )

    @staticmethod
    def get_ground_truth() -> dict[str, Any]:
        """Return the expected correct answer for this task."""
        return {
            "violation_ids": ["ART6-CONSENT"],
            "severity": "high",
            "article": "6",
        }
