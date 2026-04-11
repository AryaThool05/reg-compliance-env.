"""
tests/test_grader_guards.py

Tests that safe_score ALWAYS returns a value strictly in (0.05, 0.95)
and that all three graders never produce exact 0.0 or 1.0.

This directly validates RULE 1 of the hackathon requirements.
"""

from __future__ import annotations

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_definitions import safe_score, grade_easy, grade_medium, grade_hard
from models import RegComplianceAction


# ---------------------------------------------------------------------------
# safe_score unit tests
# ---------------------------------------------------------------------------

class TestSafeScore:
    """safe_score must always return strictly (0.05, 0.95)."""

    @pytest.mark.parametrize("raw,expected", [
        (0.0,   0.05),
        (-1.0,  0.05),
        (-100,  0.05),
        (1.0,   0.95),
        (2.0,   0.95),
        (100,   0.95),
        (0.5,   0.5),
        (0.95,  0.95),
        (0.05,  0.05),
    ])
    def test_boundary_values(self, raw: float, expected: float) -> None:
        assert safe_score(raw) == expected

    @pytest.mark.parametrize("raw", [0.1, 0.25, 0.5, 0.75, 0.9, 0.3333, 0.6667])
    def test_midrange_clamped(self, raw: float) -> None:
        result = safe_score(raw)
        assert 0.05 <= result <= 0.95, f"safe_score({raw}) = {result} out of [0.05, 0.95]"

    def test_never_returns_zero(self) -> None:
        assert safe_score(0.0) != 0.0
        assert safe_score(-999) != 0.0

    def test_never_returns_one(self) -> None:
        assert safe_score(1.0) != 1.0
        assert safe_score(999) != 1.0

    def test_strictly_less_than_one(self) -> None:
        for v in [0.0, 0.5, 1.0, -1.0, 2.0]:
            assert safe_score(v) < 1.0

    def test_strictly_greater_than_zero(self) -> None:
        for v in [0.0, 0.5, 1.0, -1.0, 2.0]:
            assert safe_score(v) > 0.0


# ---------------------------------------------------------------------------
# ground_truth fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def easy_gt():
    return {"violation_ids": ["ART6-CONSENT"], "severity": "high", "article": "6"}

@pytest.fixture
def medium_gt():
    return {"violation_ids": ["ART5-PURPOSE", "ART5-MINIMISATION", "ART6-LAWFUL-BASIS", "ART13-TRANSPARENCY", "ART5-RETENTION"], "severity": "high"}

@pytest.fixture
def hard_gt():
    return {"fixed_violations": ["ART6-LAWFUL-BASIS"], "new_violations": [], "remaining_violations": ["ART5-RETENTION", "ART13-TRANSPARENCY"], "severity": "high"}


# ---------------------------------------------------------------------------
# grade_easy guard tests
# ---------------------------------------------------------------------------

class TestGradeEasyGuards:
    """grade_easy must always return strictly (0.0, 1.0) exclusive."""

    def test_empty_action_not_zero(self, easy_gt) -> None:
        action = RegComplianceAction()
        score = grade_easy(action, easy_gt)
        assert score > 0.0, f"Expected > 0.0 but got {score}"
        assert score < 1.0, f"Expected < 1.0 but got {score}"

    def test_perfect_action_not_one(self, easy_gt) -> None:
        action = RegComplianceAction(
            violation_ids=["ART6-CONSENT"],
            severity="high",
            explanation="Lacks lawful basis",
        )
        score = grade_easy(action, easy_gt)
        assert score > 0.0, f"Expected > 0.0 but got {score}"
        assert score < 1.0, f"Expected < 1.0 but got {score}"

    def test_score_range_various_inputs(self, easy_gt) -> None:
        test_cases = [
            RegComplianceAction(),
            RegComplianceAction(violation_ids=["ART6-CONSENT"], severity="high"),
            RegComplianceAction(violation_ids=["ART5-RETENTION"], severity="low"),
            RegComplianceAction(violation_ids=["RANDOM", "GARBAGE"], severity="none"),
        ]
        for action in test_cases:
            score = grade_easy(action, easy_gt)
            assert 0.0 < score < 1.0, f"grade_easy returned {score} for {action.violation_ids}"


# ---------------------------------------------------------------------------
# grade_medium guard tests
# ---------------------------------------------------------------------------

class TestGradeMediumGuards:
    def test_empty_action_not_zero(self, medium_gt) -> None:
        action = RegComplianceAction()
        score = grade_medium(action, medium_gt)
        assert score > 0.0
        assert score < 1.0

    def test_perfect_action_not_one(self, medium_gt) -> None:
        action = RegComplianceAction(
            violation_ids=["ART5-PURPOSE", "ART5-MINIMISATION", "ART6-CONSENT", "ART13-TRANSPARENCY", "ART5-RETENTION"],
            severity="high",
        )
        score = grade_medium(action, medium_gt)
        assert score > 0.0
        assert score < 1.0

    def test_score_range_various_inputs(self, medium_gt) -> None:
        test_cases = [
            RegComplianceAction(),
            RegComplianceAction(violation_ids=["ART6-CONSENT"], severity="medium"),
            RegComplianceAction(violation_ids=["ART5-PURPOSE", "ART13-TRANSPARENCY"], severity="high"),
            RegComplianceAction(violation_ids=["JUNK1", "JUNK2", "JUNK3"], severity="low"),
        ]
        for action in test_cases:
            score = grade_medium(action, medium_gt)
            assert 0.0 < score < 1.0, f"grade_medium returned {score} for {action.violation_ids}"


# ---------------------------------------------------------------------------
# grade_hard guard tests
# ---------------------------------------------------------------------------

class TestGradeHardGuards:
    def test_empty_action_not_zero(self, hard_gt) -> None:
        action = RegComplianceAction()
        score = grade_hard(action, hard_gt)
        assert score > 0.0
        assert score < 1.0

    def test_perfect_action_not_one(self, hard_gt) -> None:
        action = RegComplianceAction(
            violation_ids=["ART6-LAWFUL-BASIS", "ART5-RETENTION", "ART13-TRANSPARENCY"],
            severity="high",
            explanation="v1 lacked lawful basis; v2 still missing retention limits and transparency.",
            fix_suggestion="Add explicit retention periods and lawful basis statement; implement erasure request workflow for Art 17 compliance.",
        )
        score = grade_hard(action, hard_gt)
        assert score > 0.0
        assert score < 1.0

    def test_score_range_various_inputs(self, hard_gt) -> None:
        test_cases = [
            RegComplianceAction(),
            RegComplianceAction(violation_ids=["ART5-RETENTION"], explanation="Short"),
            RegComplianceAction(
                violation_ids=["ART5-RETENTION", "ART13-TRANSPARENCY"],
                explanation="A" * 70,
                fix_suggestion="B" * 90,
            ),
        ]
        for action in test_cases:
            score = grade_hard(action, hard_gt)
            assert 0.0 < score < 1.0, f"grade_hard returned {score} for {action.violation_ids}"
