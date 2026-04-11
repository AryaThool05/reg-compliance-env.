"""
tests/test_environment.py

Tests for RegComplianceEnvironment reset/step/state lifecycle.
Updated to match the fixed environment:
- reset() returns RegComplianceObservation (Pydantic model, not dict)
- step() returns StepResult (with .observation, .reward, .done, .info)
- state is a property (not a method) returning RegComplianceState
"""

from __future__ import annotations

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import RegComplianceEnvironment
from models import RegComplianceAction, RegComplianceObservation, RegComplianceState


@pytest.fixture
def env() -> RegComplianceEnvironment:
    return RegComplianceEnvironment()


class TestReset:
    def test_reset_returns_pydantic_model(self, env: RegComplianceEnvironment) -> None:
        """CRITICAL: reset() must return a Pydantic model, not a dict."""
        obs = env.reset("easy")
        assert isinstance(obs, RegComplianceObservation), (
            f"reset() must return RegComplianceObservation, got {type(obs).__name__}"
        )

    def test_reset_has_model_dump(self, env: RegComplianceEnvironment) -> None:
        """openenv-core calls .model_dump() on reset() return value — must not fail."""
        obs = env.reset("easy")
        dumped = obs.model_dump()  # must not raise AttributeError
        assert isinstance(dumped, dict)
        assert "regulation_text" in dumped

    def test_reset_easy_fields(self, env: RegComplianceEnvironment) -> None:
        obs = env.reset("easy")
        assert obs.task_id == "easy"
        assert isinstance(obs.regulation_text, str)
        assert len(obs.regulation_text) > 0
        assert isinstance(obs.policy_text, str)
        assert isinstance(obs.article_refs, list)
        assert isinstance(obs.instructions, str)

    def test_reset_medium_returns_model(self, env: RegComplianceEnvironment) -> None:
        obs = env.reset("medium")
        assert isinstance(obs, RegComplianceObservation)
        assert obs.task_id == "medium"
        assert len(obs.article_refs) >= 3

    def test_reset_hard_returns_model(self, env: RegComplianceEnvironment) -> None:
        obs = env.reset("hard")
        assert isinstance(obs, RegComplianceObservation)
        assert obs.task_id == "hard"
        assert len(obs.article_refs) >= 4

    def test_reset_unknown_task_uses_fallback(self, env: RegComplianceEnvironment) -> None:
        """Unknown tasks fall back gracefully (no crash, returns valid model)."""
        obs = env.reset("impossible")
        assert isinstance(obs, RegComplianceObservation)
        assert obs.task_id in ("easy", "impossible", "hard", "medium")

    def test_reset_with_dict_task(self, env: RegComplianceEnvironment) -> None:
        """Framework may pass {"task": "easy"} — must be handled."""
        obs = env.reset({"task": "easy"})
        assert isinstance(obs, RegComplianceObservation)

    def test_reset_clears_done_flag(self, env: RegComplianceEnvironment) -> None:
        env.reset("easy")
        env.step(RegComplianceAction(violation_ids=["ART6-CONSENT"], severity="high"))
        # Re-reset should clear done
        env.reset("easy")
        state = env.state  # property access
        assert state.done is False


class TestStep:
    def test_step_returns_stepresult_with_reward(self, env: RegComplianceEnvironment) -> None:
        env.reset("easy")
        action = RegComplianceAction(violation_ids=["ART6-CONSENT"], severity="high", explanation="test")
        result = env.step(action)
        assert hasattr(result, "reward"), "StepResult must have .reward"
        assert hasattr(result, "done"), "StepResult must have .done"
        assert isinstance(result.reward, float)
        assert 0.0 < result.reward < 1.0
        assert result.done is True

    def test_step_observation_is_pydantic_model(self, env: RegComplianceEnvironment) -> None:
        """StepResult.observation must be a Pydantic model (not dict)."""
        env.reset("easy")
        result = env.step(RegComplianceAction())
        assert isinstance(result.observation, RegComplianceObservation), (
            f"step().observation must be RegComplianceObservation, got {type(result.observation).__name__}"
        )

    def test_step_observation_has_model_dump(self, env: RegComplianceEnvironment) -> None:
        env.reset("easy")
        result = env.step(RegComplianceAction())
        dumped = result.observation.model_dump()  # must not raise
        assert isinstance(dumped, dict)

    def test_step_with_dict_action(self, env: RegComplianceEnvironment) -> None:
        env.reset("easy")
        result = env.step({"violation_ids": ["ART6-CONSENT"], "severity": "high"})
        assert hasattr(result, "reward")
        assert result.done is True

    def test_step_with_none_action(self, env: RegComplianceEnvironment) -> None:
        """None action must not crash — falls back to empty action."""
        env.reset("easy")
        result = env.step(None)
        assert hasattr(result, "reward")
        assert 0.0 < result.reward < 1.0

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_all_tasks_return_valid_reward(self, task_id: str) -> None:
        env = RegComplianceEnvironment()
        env.reset(task_id)
        result = env.step(RegComplianceAction(
            violation_ids=["ART6-CONSENT", "ART5-RETENTION"],
            severity="high",
            explanation="Test violation found during audit of the privacy policy.",
            fix_suggestion="Add explicit consent mechanism and define retention periods clearly.",
        ))
        assert 0.0 < result.reward < 1.0, (
            f"Reward {result.reward} out of (0, 1) range for task {task_id}"
        )

    def test_step_reward_never_zero_or_one(self, env: RegComplianceEnvironment) -> None:
        """safe_score() guarantee: reward is strictly in (0.05, 0.95)."""
        env.reset("easy")
        result = env.step(RegComplianceAction())
        assert result.reward > 0.0, "Reward must never be exactly 0.0"
        assert result.reward < 1.0, "Reward must never be exactly 1.0"


class TestState:
    def test_state_is_property_not_method(self, env: RegComplianceEnvironment) -> None:
        """state must be a property — access without calling ()."""
        env.reset("easy")
        state = env.state  # NOT env.state()
        assert isinstance(state, RegComplianceState)

    def test_state_after_reset(self, env: RegComplianceEnvironment) -> None:
        env.reset("easy")
        state = env.state
        assert state.task_id == "easy"
        assert state.step_count == 0
        assert state.done is False

    def test_state_after_step(self, env: RegComplianceEnvironment) -> None:
        env.reset("medium")
        env.step(RegComplianceAction())
        state = env.state
        assert state.done is True
        assert state.step_count == 1

    def test_state_has_model_dump(self, env: RegComplianceEnvironment) -> None:
        env.reset("easy")
        state = env.state
        dumped = state.model_dump()
        assert isinstance(dumped, dict)
        assert "task_id" in dumped
        assert "done" in dumped
