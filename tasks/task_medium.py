"""
Medium task: full-policy audit against multiple GDPR articles.

The agent receives a full violating privacy policy and must identify
violations across Articles 5 (retention), 6 (lawful basis), and
13 (transparency).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from models import Observation

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TaskMedium:
    """Full-policy GDPR audit across Articles 5, 6, and 13."""

    DEFAULT_POLICY_PATH = str(_PROJECT_ROOT / "data" / "sample_policies" / "violating_policy.txt")

    def get_observation(
        self,
        gdpr_cache: dict[str, Any],
        policy_path: str | None = None,
    ) -> Observation:
        """Build a medium-difficulty observation.

        Combines regulation text from Articles 5, 6, and 13 with a full
        violating privacy policy loaded from disk.
        """
        path = Path(policy_path) if policy_path else Path(self.DEFAULT_POLICY_PATH)
        policy_text = path.read_text(encoding="utf-8")

        # Combine multiple article texts
        parts: list[str] = []
        for art_num in ("5", "6", "13"):
            entry = gdpr_cache.get(art_num, {})
            text = entry.get("full_text", "") if isinstance(entry, dict) else str(entry)
            if text:
                parts.append(f"=== Article {art_num} ===\n{text}")

        regulation_text = "\n\n".join(parts)

        return Observation(
            regulation_text=regulation_text,
            policy_clause=policy_text,
            task_id="medium",
            article_ref="Articles 5,6,13",
            context={
                "source": "sample_policy",
                "policy_file": path.name,
                "difficulty": "medium",
            },
        )

    @staticmethod
    def get_ground_truth() -> dict[str, Any]:
        """Return the expected violations for the violating policy."""
        return {
            "violation_ids": [
                "ART5-RETENTION",
                "ART6-LAWFUL-BASIS",
                "ART13-TRANSPARENCY",
            ],
            "severity": "high",
        }
