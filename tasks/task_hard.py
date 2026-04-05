"""
Hard task: policy-delta remediation analysis.

The agent receives two policy versions (violating → borderline) and must
identify which violations were fixed, which remain, and whether new
problems were introduced.  It must also provide a substantive fix suggestion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from models import Observation

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TaskHard:
    """Policy-delta remediation check across Articles 5–17."""

    POLICY_V1_PATH = _PROJECT_ROOT / "data" / "sample_policies" / "violating_policy.txt"
    POLICY_V2_PATH = _PROJECT_ROOT / "data" / "sample_policies" / "borderline_policy.txt"

    ARTICLES = ("5", "6", "7", "12", "13", "17")

    def get_observation(self, gdpr_cache: dict[str, Any]) -> Observation:
        """Build a hard-difficulty observation comparing two policy versions.

        Regulation text spans Articles 5, 6, 7, 12, 13, and 17.
        The policy clause contains both versions for delta analysis.
        """
        policy_v1 = self.POLICY_V1_PATH.read_text(encoding="utf-8")
        policy_v2 = self.POLICY_V2_PATH.read_text(encoding="utf-8")

        # Combine regulation texts
        parts: list[str] = []
        for art_num in self.ARTICLES:
            entry = gdpr_cache.get(art_num, {})
            text = entry.get("full_text", "") if isinstance(entry, dict) else str(entry)
            if text:
                parts.append(f"=== Article {art_num} ===\n{text}")

        regulation_text = "\n\n".join(parts)

        policy_clause = (
            f"VERSION 1:\n{policy_v1}\n\n"
            f"VERSION 2:\n{policy_v2}"
        )

        return Observation(
            regulation_text=regulation_text,
            policy_clause=policy_clause,
            task_id="hard",
            article_ref="Articles 5-17",
            context={
                "source": "sample_policy_delta",
                "policy_v1": self.POLICY_V1_PATH.name,
                "policy_v2": self.POLICY_V2_PATH.name,
                "difficulty": "hard",
            },
        )

    @staticmethod
    def get_ground_truth() -> dict[str, Any]:
        """Return the expected delta analysis between policy versions."""
        return {
            "fixed_violations": ["ART6-LAWFUL-BASIS"],
            "new_violations": [],
            "remaining_violations": ["ART5-RETENTION", "ART13-TRANSPARENCY"],
            "severity": "high",
        }
