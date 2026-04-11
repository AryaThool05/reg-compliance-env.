"""
Pydantic v2 models for the RegComplianceEnv OpenEnv environment.

Defines the three core data structures:
- RegComplianceObservation: input presented to the agent
- RegComplianceAction: the agent's compliance assessment output
- RegComplianceState: tracks the current environment state
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class RegComplianceObservation(BaseModel):
    """Represents a single compliance-checking task presented to the agent.

    Contains the GDPR regulation text alongside the company policy text
    that must be evaluated for compliance violations.
    """

    regulation_text: str = Field(
        ..., description="GDPR article text to check against."
    )
    policy_text: str = Field(
        ..., description="The company policy text to evaluate for compliance."
    )
    task_id: Literal["easy", "medium", "hard"] = Field(
        ..., description="Difficulty tier of this compliance task."
    )
    article_refs: list[str] = Field(
        ..., description='GDPR article references, e.g. ["Article 6"].'
    )
    instructions: str = Field(
        ..., description="Explicit task instructions for the LLM agent."
    )
    context: dict = Field(
        default_factory=dict,
        description="Extra metadata (e.g. difficulty, source).",
    )

    def to_prompt(self) -> str:
        """Return a clean LLM prompt string. No trailing newlines in action output."""
        articles = ", ".join(self.article_refs)
        return (
            f"TASK INSTRUCTIONS:\n{self.instructions}\n\n"
            f"GDPR REFERENCE ({articles}):\n{self.regulation_text[:3000]}\n\n"
            f"POLICY TO AUDIT:\n{self.policy_text[:2000]}\n\n"
            "Respond with ONLY a JSON object — no markdown, no explanation outside JSON:\n"
            '{"violation_ids": ["ART6-CONSENT"], "severity": "high", '
            '"explanation": "...", "fix_suggestion": "..."}'
        )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class RegComplianceAction(BaseModel):
    """The agent's compliance assessment for a given observation."""

    violation_ids: list[str] = Field(
        default_factory=list,
        description='List of violation IDs, e.g. ["ART6-CONSENT", "ART5-RETENTION"].',
    )
    severity: Literal["none", "low", "medium", "high"] = Field(
        default="none",
        description="Overall severity of the identified violations.",
    )
    explanation: str = Field(
        default="",
        description="Brief explanation of the compliance assessment (under 200 chars).",
    )
    fix_suggestion: str = Field(
        default="",
        description="Concrete suggestion to remediate the violations.",
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RegComplianceState(BaseModel):
    """Tracks the current state of the RegComplianceEnv environment."""

    task_id: str = Field(default="easy", description="Active task identifier.")
    step_count: int = Field(default=0, description="Steps taken in current episode.")
    done: bool = Field(default=False, description="Whether the episode has ended.")
    episode_id: str = Field(default="", description="Unique episode identifier.")


# ---------------------------------------------------------------------------
# Backward-compat aliases (for old code that uses Observation / Action)
# ---------------------------------------------------------------------------

Observation = RegComplianceObservation
Action = RegComplianceAction
