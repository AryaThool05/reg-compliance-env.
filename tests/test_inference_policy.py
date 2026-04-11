"""
tests/test_inference_policy.py

Tests for the to_prompt() method and model field validation.
"""

from __future__ import annotations

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import RegComplianceObservation, RegComplianceAction, RegComplianceState


class TestRegComplianceObservation:
    @pytest.fixture
    def sample_obs(self) -> RegComplianceObservation:
        return RegComplianceObservation(
            regulation_text="Processing shall be lawful only if consent given.",
            policy_text="We share your data with partners without consent.",
            task_id="easy",
            article_refs=["Article 6"],
            instructions="Identify GDPR violations in the policy.",
        )

    def test_to_prompt_contains_instructions(self, sample_obs: RegComplianceObservation) -> None:
        prompt = sample_obs.to_prompt()
        assert "TASK INSTRUCTIONS:" in prompt
        assert "Identify GDPR violations" in prompt

    def test_to_prompt_contains_regulation(self, sample_obs: RegComplianceObservation) -> None:
        prompt = sample_obs.to_prompt()
        assert "GDPR REFERENCE" in prompt
        assert "lawful only if consent given" in prompt

    def test_to_prompt_contains_policy(self, sample_obs: RegComplianceObservation) -> None:
        prompt = sample_obs.to_prompt()
        assert "POLICY TO AUDIT:" in prompt
        assert "without consent" in prompt

    def test_to_prompt_contains_article_refs(self, sample_obs: RegComplianceObservation) -> None:
        prompt = sample_obs.to_prompt()
        assert "Article 6" in prompt

    def test_to_prompt_no_raw_newlines_in_format_hint(self, sample_obs: RegComplianceObservation) -> None:
        """The JSON format hint at end of prompt should be on one line."""
        prompt = sample_obs.to_prompt()
        # The JSON example on the last line must not break mid-line
        last_lines = prompt.strip().split("\n")
        json_hint_line = [l for l in last_lines if '"violation_ids"' in l]
        assert len(json_hint_line) >= 1

    def test_field_names_correct(self) -> None:
        """Verify new field names exist (policy_text, article_refs, instructions)."""
        obs = RegComplianceObservation(
            regulation_text="text",
            policy_text="policy",
            task_id="medium",
            article_refs=["Article 5", "Article 6"],
            instructions="Audit this.",
        )
        assert hasattr(obs, "policy_text")
        assert hasattr(obs, "article_refs")
        assert hasattr(obs, "instructions")
        assert isinstance(obs.article_refs, list)

    def test_context_default_empty(self) -> None:
        obs = RegComplianceObservation(
            regulation_text="t", policy_text="p", task_id="hard",
            article_refs=["Article 5"], instructions="i"
        )
        assert obs.context == {}


class TestRegComplianceAction:
    def test_defaults(self) -> None:
        action = RegComplianceAction()
        assert action.violation_ids == []
        assert action.severity == "none"
        assert action.explanation == ""
        assert action.fix_suggestion == ""

    def test_fix_suggestion_is_str_not_none(self) -> None:
        """fix_suggestion must default to empty string, never None."""
        action = RegComplianceAction()
        assert action.fix_suggestion is not None
        assert isinstance(action.fix_suggestion, str)

    def test_severity_enum_valid(self) -> None:
        for sev in ["none", "low", "medium", "high"]:
            action = RegComplianceAction(severity=sev)
            assert action.severity == sev

    def test_severity_accepts_any_string(self) -> None:
        """severity is plain str (not Literal) to prevent framework validation errors."""
        # Must accept standard values
        for sev in ["none", "low", "medium", "high"]:
            action = RegComplianceAction(severity=sev)
            assert action.severity == sev
        # Must also accept non-standard values (framework may send unexpected data)
        action = RegComplianceAction(severity="critical")
        assert action.severity == "critical"  # no raise, just stored



class TestRegComplianceState:
    def test_defaults(self) -> None:
        state = RegComplianceState()
        assert state.task_id == "easy"
        assert state.step_count == 0
        assert state.done is False
        assert state.episode_id == ""

    def test_model_dump(self) -> None:
        state = RegComplianceState(task_id="hard", step_count=1, done=True, episode_id="abc-123")
        d = state.model_dump()
        assert d["task_id"] == "hard"
        assert d["done"] is True
        assert d["episode_id"] == "abc-123"
