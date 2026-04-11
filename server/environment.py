"""
server/environment.py — RegComplianceEnvironment

CRITICAL FIX: reset() MUST return a RegComplianceObservation (Pydantic model),
NOT a dict. openenv-core calls .model_dump() on the return value of reset(),
so returning a plain dict causes:
    AttributeError: 'dict' object has no attribute 'model_dump'

Rules for all return types:
- reset()  → RegComplianceObservation (Pydantic instance, NEVER dict)
- step()   → StepResult with .observation = RegComplianceObservation instance
- state    → RegComplianceState (Pydantic instance)
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

# ---------------------------------------------------------------------------
# openenv.core imports — with detailed fallback
# ---------------------------------------------------------------------------

try:
    from openenv.core.env_server import Environment, StepResult
except ImportError:
    try:
        from openenv.core import Environment, StepResult  # type: ignore[no-redef]
    except ImportError:
        # Minimal standalone fallbacks when openenv-core is not installed
        class Environment:  # type: ignore[no-redef]
            """Minimal base when openenv-core is absent."""
            pass

        class StepResult:  # type: ignore[no-redef]
            """Minimal StepResult for standalone mode."""
            def __init__(self, observation, reward, done, info=None):
                self.observation = observation
                self.reward = reward
                self.done = done
                self.info = info or {}

# ---------------------------------------------------------------------------
# Model imports — dual pattern (package or standalone)
# ---------------------------------------------------------------------------

try:
    from ..models import RegComplianceObservation, RegComplianceAction, RegComplianceState
    from ..task_definitions import TASK_CONFIGS, get_task_config
except ImportError:
    from models import RegComplianceObservation, RegComplianceAction, RegComplianceState
    from task_definitions import TASK_CONFIGS, get_task_config

# ---------------------------------------------------------------------------
# Path resolution for static GDPR JSON
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GDPR_JSON_PATH = os.path.join(_BASE_DIR, "data", "gdpr_articles.json")


def safe_score(s: float) -> float:
    """Clamp score strictly to (0.05, 0.95) — never returns 0.0 or 1.0."""
    return max(0.05, min(0.95, float(s)))


class RegComplianceEnvironment(Environment):
    """GDPR compliance-checking environment compatible with openenv-core.

    RETURN TYPE CONTRACT (strictly enforced):
    - reset()  → RegComplianceObservation  (Pydantic model, NEVER dict)
    - step()   → StepResult where .observation is RegComplianceObservation
    - state    → RegComplianceState         (Pydantic model)
    """

    def __init__(self) -> None:
        self._current_task: str = "easy"
        self._step_count: int = 0
        self._done: bool = False
        self._episode_id: str = ""
        self._observation: RegComplianceObservation = RegComplianceObservation()
        self._ground_truth: dict[str, Any] = {}

    # ---- Static GDPR data -------------------------------------------------

    def _load_gdpr_articles(self) -> dict[str, Any]:
        """Load GDPR article text from static JSON. Never raises."""
        try:
            with open(_GDPR_JSON_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return self._get_static_fallback()

    def _get_static_fallback(self) -> dict[str, Any]:
        """Hardcoded minimal GDPR text — the safety net, never raises, never reads files."""
        return {
            "5": {
                "full_text": (
                    "Article 5 — Principles: Personal data shall be processed lawfully, "
                    "fairly and transparently. It must be collected for specified, explicit "
                    "and legitimate purposes (purpose limitation), adequate and limited to "
                    "what is necessary (data minimisation), accurate, kept no longer than "
                    "necessary (storage limitation), and processed securely."
                )
            },
            "6": {
                "full_text": (
                    "Article 6 — Lawfulness of processing: Processing is lawful only if at "
                    "least one applies: (a) consent given; (b) contract necessity; "
                    "(c) legal obligation; (d) vital interests; (e) public task; "
                    "(f) legitimate interests. Without a valid legal basis, processing is unlawful."
                )
            },
            "13": {
                "full_text": (
                    "Article 13 — Transparency: At collection, the controller must provide: "
                    "identity and contact details, purposes and legal basis, recipients of data, "
                    "retention period, data subject rights, right to withdraw consent, "
                    "and right to lodge a complaint with a supervisory authority."
                )
            },
            "17": {
                "full_text": (
                    "Article 17 — Right to erasure: Data subjects have the right to obtain "
                    "erasure of personal data without undue delay where the data is no longer "
                    "necessary, consent is withdrawn, or processing is unlawful."
                )
            },
        }

    def _build_fallback_observation(self, task_id: str = "easy") -> RegComplianceObservation:
        """Construct a safe default observation using hardcoded text. Never raises."""
        gdpr = self._get_static_fallback()
        art6_text = gdpr["6"]["full_text"]
        return RegComplianceObservation(
            regulation_text=art6_text,
            policy_text=(
                "We share your personal data with marketing partners without "
                "requiring your explicit consent. Users cannot opt out of this sharing."
            ),
            task_id=task_id if task_id in ("easy", "medium", "hard") else "easy",
            article_refs=["Article 6"],
            instructions=(
                "Identify GDPR violations in the policy clause. "
                "Check whether the policy has a valid lawful basis under Article 6. "
                "Return violation IDs like ART6-CONSENT or ART6-LAWFUL-BASIS."
            ),
            context={"task": task_id, "source": "fallback"},
        )

    # ---- OpenEnv interface (STRICTLY TYPED RETURNS) -----------------------

    def reset(self, task: Any = "easy", **kwargs: Any) -> RegComplianceObservation:
        """Reset the environment for a new episode.

        RETURNS: RegComplianceObservation (Pydantic model — NEVER a dict).
        openenv-core calls .model_dump() on this return value.

        Args:
            task: Task ID string or dict with "task" key. Defaults to "easy".
        """
        try:
            # Normalise task argument — framework may pass a dict like {"task": "easy"}
            if isinstance(task, dict):
                task_id = str(task.get("task", task.get("task_id", "easy")))
            elif isinstance(task, str):
                task_id = task
            else:
                task_id = str(task) if task else "easy"

            # Validate and default
            if task_id not in ("easy", "medium", "hard"):
                task_id = "easy"

            # Load GDPR data (static JSON, no network)
            gdpr = self._load_gdpr_articles()

            # Build observation using task_definitions
            config = get_task_config(task_id)
            observation = config["build_observation"](gdpr)  # returns RegComplianceObservation

            # Store state
            self._current_task = task_id
            self._step_count = 0
            self._done = False
            self._episode_id = str(uuid.uuid4())
            self._observation = observation
            self._ground_truth = config["ground_truth"]()

            # MUST return a Pydantic model instance, never a dict
            return observation

        except Exception as exc:
            # Safe fallback — never let reset() raise; always return a valid Pydantic model
            self._current_task = "easy"
            self._step_count = 0
            self._done = False
            self._episode_id = str(uuid.uuid4())
            fallback = self._build_fallback_observation("easy")
            self._observation = fallback
            self._ground_truth = {}
            return fallback

    def step(self, action: Any = None, **kwargs: Any) -> StepResult:
        """Execute one environment step.

        RETURNS: StepResult where .observation is RegComplianceObservation (Pydantic model).

        Args:
            action: RegComplianceAction instance or dict with action fields.
        """
        try:
            # Normalise action — may arrive as a dict from the HTTP layer
            if action is None:
                act = RegComplianceAction()
            elif isinstance(action, dict):
                act = RegComplianceAction(**{
                    k: v for k, v in action.items()
                    if k in RegComplianceAction.model_fields
                })
            elif isinstance(action, RegComplianceAction):
                act = action
            else:
                act = RegComplianceAction()

            # Grade using the appropriate grader
            config = TASK_CONFIGS.get(self._current_task, TASK_CONFIGS["easy"])
            raw_score = config["grader"](act, self._ground_truth)
            reward = safe_score(raw_score)

            # Update step state
            self._step_count += 1
            self._done = True

            # MUST return StepResult with Pydantic model as observation
            return StepResult(
                observation=self._observation,  # RegComplianceObservation instance
                reward=reward,
                done=True,
                info={
                    "task": self._current_task,
                    "score": reward,
                    "step_count": self._step_count,
                    "episode_id": self._episode_id,
                },
            )

        except Exception as exc:
            # Safe fallback — never crash; return a minimal valid StepResult
            self._done = True
            return StepResult(
                observation=self._observation,  # still a Pydantic model
                reward=0.05,
                done=True,
                info={"task": self._current_task, "error": str(exc)[:200]},
            )

    @property
    def state(self) -> RegComplianceState:
        """Return current environment state as a RegComplianceState Pydantic model."""
        return RegComplianceState(
            task_id=self._current_task,
            step_count=self._step_count,
            done=self._done,
            episode_id=self._episode_id,
        )
