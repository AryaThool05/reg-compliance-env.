"""
RegComplianceEnv — an async OpenEnv environment for GDPR compliance checking.

Provides ``reset``, ``step``, ``state``, and ``close`` as fully async methods,
compatible with asyncio-based inference scripts and the OpenEnv specification.

Memory target: < 200 MB RAM for the entire environment lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from models import Action, EnvState, Observation, Reward
from scraper import load_gdpr_cache
from tasks import TaskEasy, TaskMedium, TaskHard
from graders import grade_easy, grade_medium, grade_hard


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ResetResult:
    """Returned by ``RegComplianceEnv.reset()``."""

    observation: Observation
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Returned by ``RegComplianceEnv.step()``."""

    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Task & grader registries
# ---------------------------------------------------------------------------

TASK_MAP: dict[str, type] = {
    "easy": TaskEasy,
    "medium": TaskMedium,
    "hard": TaskHard,
}

_GRADER_MAP = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


# ---------------------------------------------------------------------------
# Reward computation helper
# ---------------------------------------------------------------------------

def compute_reward(action: Action, ground_truth: dict[str, Any], task_id: str) -> Reward:
    """Compute a reward with partial credit, dispatching to the right grader.

    Fallback scoring (when no grader matches) uses a generic rubric:
      +0.10  any violations found
      +0.30  × precision
      +0.30  × recall
      +0.20  × severity accuracy
      −0.15  if > 5 false positives
    Result is clamped to [0.0, 1.0].
    """
    grader = _GRADER_MAP.get(task_id)
    if grader is not None:
        return grader(action, ground_truth)

    # ----- Generic fallback scoring -----
    gold_ids = set(ground_truth.get("violation_ids", []))
    predicted = set(action.violation_ids)
    tp = gold_ids & predicted
    fp = predicted - gold_ids

    score = 0.0

    # +0.1 if any violations found
    if predicted:
        score += 0.1

    # Precision & recall
    precision = len(tp) / len(predicted) if predicted else 0.0
    recall = len(tp) / len(gold_ids) if gold_ids else 0.0
    score += 0.3 * precision
    score += 0.3 * recall

    # Severity accuracy
    expected_severity = ground_truth.get("severity", "")
    severity_match = 1.0 if action.severity == expected_severity else 0.0
    score += 0.2 * severity_match

    # False-positive penalty
    if len(fp) > 5:
        score -= 0.15

    score = max(0.05, min(0.95, score))

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.05

    return Reward(
        score=score,
        precision=precision,
        recall=recall,
        f1=f1,
        breakdown={
            "any_violation_bonus": 0.1 if predicted else 0.0,
            "precision_component": 0.3 * precision,
            "recall_component": 0.3 * recall,
            "severity_component": 0.2 * severity_match,
            "fp_penalty": -0.15 if len(fp) > 5 else 0.0,
        },
    )


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class RegComplianceEnv:
    """Async GDPR compliance-checking environment (OpenEnv compatible).

    Usage::

        env = RegComplianceEnv()
        result = await env.reset(task="easy")
        prompt = result.observation.to_prompt()
        # … call LLM …
        step_result = await env.step(action)
        print(step_result.reward)
        await env.close()
    """

    def __init__(self) -> None:
        self._state: EnvState | None = None
        self._gdpr_cache: dict[str, Any] = {}
        self._observation: Observation | None = None
        self._ground_truth: dict[str, Any] = {}
        self._task_instance: Any = None

    # ---- async API --------------------------------------------------------

    async def reset(self, task: str = "easy") -> ResetResult:
        """Reset the environment for a new episode.

        Args:
            task: Difficulty tier — ``"easy"``, ``"medium"``, or ``"hard"``.

        Returns:
            A :class:`ResetResult` containing the initial observation.
        """
        if task not in TASK_MAP:
            raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_MAP)}")

        # Load GDPR cache from disk (no network calls)
        self._gdpr_cache = load_gdpr_cache()

        # Instantiate the task and build the observation
        task_cls = TASK_MAP[task]
        self._task_instance = task_cls()
        self._observation = self._task_instance.get_observation(self._gdpr_cache)
        self._ground_truth = self._task_instance.get_ground_truth()

        # Initialise environment state
        self._state = EnvState(
            current_task=task,
            step_count=0,
            done=False,
            last_action=None,
            regulation_loaded=bool(self._gdpr_cache),
            policy_loaded=True,
        )

        return ResetResult(
            observation=self._observation,
            info={
                "task": task,
                "article_ref": self._observation.article_ref,
                "cache_articles": list(self._gdpr_cache.keys()),
            },
        )

    async def step(self, action: Action) -> StepResult:
        """Execute one step: grade the agent's action and return a reward.

        Each episode is single-step — ``done`` is always ``True`` after step.

        Args:
            action: The agent's compliance assessment.

        Returns:
            A :class:`StepResult` with reward and done=True.

        Raises:
            TypeError: If *action* is not an :class:`Action` instance.
            RuntimeError: If called before ``reset()`` or after episode is done.
        """
        if not isinstance(action, Action):
            raise TypeError(f"Expected Action instance, got {type(action).__name__}")

        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        if self._state.done:
            raise RuntimeError("Episode is already done. Call reset() to start a new one.")

        # Compute reward
        reward: Reward = compute_reward(
            action, self._ground_truth, self._state.current_task
        )

        # Update state
        self._state.step_count += 1
        self._state.done = True
        self._state.last_action = action

        return StepResult(
            observation=self._observation,  # type: ignore[arg-type]
            reward=reward.score,
            done=True,
            info={
                "reward_details": reward.model_dump(),
                "ground_truth_ids": self._ground_truth.get(
                    "violation_ids",
                    self._ground_truth.get("remaining_violations", []),
                ),
            },
        )

    async def state(self) -> dict[str, Any]:
        """Return the current environment state as a plain dict."""
        if self._state is None:
            return {"error": "Environment not initialised. Call reset() first."}
        return self._state.model_dump()

    async def close(self) -> None:
        """Release any held resources.

        Currently lightweight — clears internal references to help the GC.
        """
        self._gdpr_cache = {}
        self._observation = None
        self._ground_truth = {}
        self._task_instance = None
        self._state = None
