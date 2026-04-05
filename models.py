"""
Pydantic v2 models for the RegComplianceEnv OpenEnv environment.

Defines the core data structures used for GDPR compliance checking:
- Observation: input presented to the agent (regulation + policy text)
- Action: the agent's compliance assessment output
- Reward: scored evaluation of the agent's action
- EnvState: tracks the current state of the environment
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Represents a single compliance-checking task presented to the agent.

    Contains the scraped GDPR regulation text alongside the company policy
    clause that must be evaluated for compliance violations.
    """

    regulation_text: str = Field(
        ..., description="The scraped GDPR article text to check against."
    )
    policy_clause: str = Field(
        ..., description="The company policy text to evaluate for compliance."
    )
    task_id: Literal["easy", "medium", "hard"] = Field(
        ..., description="Difficulty tier of this compliance task."
    )
    article_ref: str = Field(
        ..., description='GDPR article reference, e.g. "Article 6".'
    )
    context: dict = Field(
        default_factory=dict,
        description="Extra metadata (e.g. jurisdiction, sector, timestamps).",
    )

    def to_prompt(self) -> str:
        """Return a clean, human-readable prompt string for LLM input.

        The prompt is formatted so the model can clearly distinguish the
        regulation from the policy clause and understand its task.
        """
        lines = [
            f"=== GDPR Compliance Check ({self.article_ref}) ===",
            f"Difficulty: {self.task_id}",
            "",
            "--- Regulation Text ---",
            self.regulation_text.strip(),
            "",
            "--- Company Policy Clause ---",
            self.policy_clause.strip(),
            "",
            "Analyze the policy clause above for potential GDPR violations "
            "against the referenced regulation. Identify specific violation "
            "IDs, assess severity, provide an explanation, and suggest fixes.",
        ]
        if self.context:
            lines.insert(3, f"Context: {self.context}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """The agent's compliance assessment for a given observation.

    Lists identified violations, their aggregate severity, an explanation,
    and an optional fix suggestion.
    """

    violation_ids: list[str] = Field(
        default_factory=list,
        description='List of violation identifiers, e.g. ["ART6-1", "ART13-2"].',
    )
    severity: Literal["none", "low", "medium", "high"] = Field(
        ..., description="Overall severity of the identified violations."
    )
    explanation: str = Field(
        ..., description="Free-text explanation of the compliance assessment."
    )
    fix_suggestion: str | None = Field(
        default=None,
        description="Optional suggestion for how to remediate the violation(s).",
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Scored evaluation of the agent's action against the ground-truth labels.

    The primary ``score`` is clamped to ``[0.0, 1.0]``.  Additional metrics
    (precision, recall, F1) allow fine-grained analysis, and ``breakdown``
    stores per-criterion scores.
    """

    score: float = Field(
        ..., description="Overall score clamped to [0.0, 1.0]."
    )
    precision: float = Field(
        ..., description="Precision of violation identification."
    )
    recall: float = Field(
        ..., description="Recall of violation identification."
    )
    f1: float = Field(
        ..., description="F1 score combining precision and recall."
    )
    breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Per-criterion score breakdown (e.g. severity accuracy, explanation quality).",
    )

    @field_validator("score")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """Ensure score is always within [0.0, 1.0]."""
        return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# EnvState
# ---------------------------------------------------------------------------

class EnvState(BaseModel):
    """Tracks the current state of the RegComplianceEnv environment.

    Used internally to manage the lifecycle of a compliance-checking episode,
    including which task is active, how many steps have been taken, and
    whether required data has been loaded.
    """

    current_task: str = Field(
        ..., description="Identifier of the currently active task."
    )
    step_count: int = Field(
        default=0, description="Number of steps taken in the current episode."
    )
    done: bool = Field(
        default=False, description="Whether the current episode has concluded."
    )
    last_action: Action | None = Field(
        default=None, description="The most recent action taken by the agent."
    )
    regulation_loaded: bool = Field(
        default=False, description="Whether GDPR regulation data has been loaded."
    )
    policy_loaded: bool = Field(
        default=False, description="Whether company policy data has been loaded."
    )
