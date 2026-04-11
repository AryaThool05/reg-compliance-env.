"""
Pydantic v2 models for the RegComplianceEnv OpenEnv environment.

ALL fields have default values so models can be instantiated with zero
arguments. This prevents crashes in fallback/error paths.

reward + done added to RegComplianceObservation to satisfy any openenv.core
serialization that calls observation.reward / observation.done.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observation — returned by reset()
# ---------------------------------------------------------------------------

class RegComplianceObservation(BaseModel):
    """Represents a single compliance-checking task presented to the agent.

    ALL fields have defaults so this can safely be constructed with zero
    arguments in error/fallback paths.

    reward and done are included to satisfy openenv.core serialize_observation()
    which may call observation.reward and observation.done.
    """

    regulation_text: str = ""
    policy_text: str = ""
    task_id: str = "easy"
    article_refs: list[str] = Field(default_factory=list)
    instructions: str = ""
    context: dict = Field(default_factory=dict)
    # Satisfy openenv.core serialization (calls observation.reward / observation.done)
    reward: float = 0.5
    done: bool = False

    def to_prompt(self) -> str:
        """Return a clean LLM prompt string."""
        articles = ", ".join(self.article_refs) if self.article_refs else "GDPR"
        return (
            f"TASK INSTRUCTIONS:\n{self.instructions}\n\n"
            f"GDPR REFERENCE ({articles}):\n{self.regulation_text[:3000]}\n\n"
            f"POLICY TO AUDIT:\n{self.policy_text[:2000]}\n\n"
            "Respond with ONLY a JSON object — no markdown, no explanation outside JSON:\n"
            '{"violation_ids": ["ART6-CONSENT"], "severity": "high", '
            '"explanation": "...", "fix_suggestion": "..."}'
        )


# ---------------------------------------------------------------------------
# Action — submitted to step()
# ---------------------------------------------------------------------------

class RegComplianceAction(BaseModel):
    """The agent's compliance assessment for a given observation.

    ALL fields have defaults. severity is plain str (not Literal) to avoid
    validation failures from unexpected values sent by the framework.
    """

    violation_ids: list[str] = Field(default_factory=list)
    severity: str = "none"
    explanation: str = ""
    fix_suggestion: str = ""


# ---------------------------------------------------------------------------
# State — returned by state property
# ---------------------------------------------------------------------------

class RegComplianceState(BaseModel):
    """Tracks the current state of the RegComplianceEnv environment."""

    task_id: str = "easy"
    step_count: int = 0
    done: bool = False
    episode_id: str = ""


# ---------------------------------------------------------------------------
# Backward-compat aliases
# ---------------------------------------------------------------------------

Observation = RegComplianceObservation
Action = RegComplianceAction
