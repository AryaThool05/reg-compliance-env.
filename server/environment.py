"""
server/environment.py — RegComplianceEnvironment

Implements the OpenEnv Environment interface for GDPR compliance checking.
Loads GDPR articles from static JSON (no network calls at runtime).

Compatible with openenv.core.Environment if installed; falls back to a
standalone base class otherwise so local runs always work.
"""

from __future__ import annotations

import uuid
from typing import Any

try:
    from openenv.core import Environment as _BaseEnvironment
except ImportError:
    # Standalone fallback base class — matches the interface judges expect
    class _BaseEnvironment:  # type: ignore[no-redef]
        """Minimal base class when openenv-core is not installed."""
        pass

try:
    from ..models import RegComplianceObservation, RegComplianceAction, RegComplianceState
    from ..task_definitions import TASK_CONFIGS, load_gdpr_articles, get_task_config
except ImportError:
    from models import RegComplianceObservation, RegComplianceAction, RegComplianceState
    from task_definitions import TASK_CONFIGS, load_gdpr_articles, get_task_config


class RegComplianceEnvironment(_BaseEnvironment):
    """GDPR compliance-checking environment (OpenEnv compatible).

    Each episode is single-step:
      1. reset(task_id) → observation
      2. step(action)   → reward (float in (0.05, 0.95))
      3. done = True

    No network calls at runtime — all GDPR data from static JSON.
    """

    # Class-level GDPR cache — loaded once, shared across all instances
    _gdpr_cache: dict[str, Any] | None = None

    def __init__(self) -> None:
        self._state: RegComplianceState | None = None
        self._observation: RegComplianceObservation | None = None
        self._ground_truth: dict[str, Any] = {}
        self._task_id: str = "easy"

    # ---- GDPR data loading ------------------------------------------------

    @classmethod
    def _get_gdpr_cache(cls) -> dict[str, Any]:
        """Load GDPR articles once and cache at class level."""
        if cls._gdpr_cache is None:
            cls._gdpr_cache = load_gdpr_articles()
        return cls._gdpr_cache

    # ---- OpenEnv interface ------------------------------------------------

    def reset(self, task_id: str = "easy", **kwargs: Any) -> dict[str, Any]:
        """Reset environment for a new episode.

        Args:
            task_id: One of "easy", "medium", "hard".

        Returns:
            Observation dict compatible with OpenEnv spec.
        """
        config = get_task_config(task_id)
        gdpr = self._get_gdpr_cache()

        self._task_id = task_id
        self._observation = config["build_observation"](gdpr)
        self._ground_truth = config["ground_truth"]()
        self._state = RegComplianceState(
            task_id=task_id,
            step_count=0,
            done=False,
            episode_id=str(uuid.uuid4()),
        )

        return self._observation.model_dump()

    def step(self, action: dict[str, Any] | RegComplianceAction, **kwargs: Any) -> dict[str, Any]:
        """Execute one step: grade the agent's action and return reward.

        Args:
            action: Action dict or RegComplianceAction instance.

        Returns:
            Dict with keys: observation, reward, done, info.
        """
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        if self._state.done:
            raise RuntimeError("Episode already done. Call reset() to start a new one.")

        # Normalise to RegComplianceAction
        if isinstance(action, dict):
            act = RegComplianceAction(**action)
        elif isinstance(action, RegComplianceAction):
            act = action
        else:
            raise TypeError(f"Expected dict or RegComplianceAction, got {type(action).__name__}")

        # Grade
        config = TASK_CONFIGS[self._task_id]
        reward: float = config["grader"](act, self._ground_truth)

        # Update state
        self._state.step_count += 1
        self._state.done = True

        return {
            "observation": self._observation.model_dump() if self._observation else {},
            "reward": reward,
            "done": True,
            "info": {
                "task_id": self._task_id,
                "step_count": self._state.step_count,
                "episode_id": self._state.episode_id,
                "violation_ids_submitted": act.violation_ids,
            },
        }

    def state(self) -> dict[str, Any]:
        """Return current environment state as a plain dict."""
        if self._state is None:
            return {"error": "Environment not initialised. Call reset() first."}
        return self._state.model_dump()
