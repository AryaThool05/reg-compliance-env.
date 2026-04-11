"""
tests/test_environment.py

Tests for RegComplianceEnvironment reset/step/state lifecycle.
"""

from __future__ import annotations

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import RegComplianceEnvironment
from models import RegComplianceAction


@pytest.fixture
def env() -> RegComplianceEnvironment:
    return RegComplianceEnvironment()


class TestReset:
    def test_reset_easy_returns_observation(self, env: RegComplianceEnvironment) -> None:
        obs = env.reset("easy")
        assert isinstance(obs, dict)
        assert "regulation_text" in obs
        assert "policy_text" in obs
        assert "task_id" in obs
        assert obs["task_id"] == "easy"
        assert "article_refs" in obs
        assert "instructions" in obs

    def test_reset_medium_returns_observation(self, env: RegComplianceEnvironment) -> None:
        obs = env.reset("medium")
        assert obs["task_id"] == "medium"
        assert len(obs["article_refs"]) >= 3

    def test_reset_hard_returns_observation(self, env: RegComplianceEnvironment) -> None:
        obs = env.reset("hard")
        assert obs["task_id"] == "hard"
        assert len(obs["article_refs"]) >= 6

    def test_reset_unknown_task_raises(self, env: RegComplianceEnvironment) -> None:
        with pytest.raises(ValueError, match="Unknown task"):
            env.reset("impossible")

    def test_reset_clears_done_flag(self, env: RegComplianceEnvironment) -> None:
        env.reset("easy")
        action = RegComplianceAction(violation_ids=["ART6-CONSENT"], severity="high")
        env.step(action)
        # Re-reset should clear done
        env.reset("easy")
        state = env.state()
        assert state["done"] is False


class TestStep:
    def test_step_returns_floats_in_range(self, env: RegComplianceEnvironment) -> None:
        env.reset("easy")
        action = RegComplianceAction(violation_ids=["ART6-CONSENT"], severity="high", explanation="test")
        result = env.step(action)
        assert isinstance(result["reward"], float)
        assert 0.0 < result["reward"] < 1.0
        assert result["done"] is True

    def test_step_with_dict_action(self, env: RegComplianceEnvironment) -> None:
        env.reset("easy")
        result = env.step({"violation_ids": ["ART6-CONSENT"], "severity": "high"})
        assert "reward" in result
        assert result["done"] is True

    def test_step_without_reset_raises(self, env: RegComplianceEnvironment) -> None:
        with pytest.raises(RuntimeError, match="not initialised"):
            env.step(RegComplianceAction())

    def test_step_twice_raises(self, env: RegComplianceEnvironment) -> None:
        env.reset("easy")
        env.step(RegComplianceAction())
        with pytest.raises(RuntimeError, match="already done"):
            env.step(RegComplianceAction())

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_all_tasks_return_valid_reward(self, task_id: str) -> None:
        env = RegComplianceEnvironment()
        env.reset(task_id)
        result = env.step(RegComplianceAction(
            violation_ids=["ART6-CONSENT", "ART5-RETENTION"],
            severity="high",
            explanation="Test violation found during audit.",
            fix_suggestion="Add explicit consent mechanism and retention policy.",
        ))
        assert 0.0 < result["reward"] < 1.0, f"Reward {result['reward']} out of range for task {task_id}"


class TestState:
    def test_state_before_reset(self, env: RegComplianceEnvironment) -> None:
        state = env.state()
        assert "error" in state

    def test_state_after_reset(self, env: RegComplianceEnvironment) -> None:
        env.reset("easy")
        state = env.state()
        assert state["task_id"] == "easy"
        assert state["step_count"] == 0
        assert state["done"] is False

    def test_state_after_step(self, env: RegComplianceEnvironment) -> None:
        env.reset("medium")
        env.step(RegComplianceAction())
        state = env.state()
        assert state["done"] is True
        assert state["step_count"] == 1
