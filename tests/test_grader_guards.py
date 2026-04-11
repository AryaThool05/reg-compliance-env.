"""
tests/test_grader_guards.py

Mirrors the EXACT Phase 2 validator check:
  "Each task's score must be strictly between 0 and 1 (not 0.0 and not 1.0)"

Also tests safe_score() for boundary safety and float precision edge cases.
"""

from __future__ import annotations

import sys
import os
import math
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import RegComplianceEnvironment, safe_score
from models import RegComplianceAction


# ---------------------------------------------------------------------------
# Shared environment instance and test actions
# ---------------------------------------------------------------------------

env = RegComplianceEnvironment()

ACTIONS_TO_TEST = [
    # Perfect answer — max score path
    RegComplianceAction(
        violation_ids=["ART6-CONSENT", "ART5-RETENTION", "ART13-TRANSPARENCY"],
        severity="high",
        explanation="Policy lacks explicit consent mechanism and does not specify a retention period. Third-party sharing without lawful basis.",
        fix_suggestion="Add consent checkbox before data collection. Define 2-year retention policy. Document all third-party processors and their legal basis.",
    ),
    # Empty answer — minimum score path, must still be > 0.0
    RegComplianceAction(
        violation_ids=[],
        severity="none",
        explanation="",
        fix_suggestion="",
    ),
    # Partial — mid-range score
    RegComplianceAction(
        violation_ids=["ART6-CONSENT"],
        severity="medium",
        explanation="Missing consent",
        fix_suggestion="Add consent",
    ),
    # Wrong violation IDs — no keyword matches
    RegComplianceAction(
        violation_ids=["FAKE-VIOLATION", "ANOTHER-FAKE"],
        severity="low",
        explanation="x",
        fix_suggestion="y",
    ),
]


# ---------------------------------------------------------------------------
# Phase 2 core check: 3 tasks × 4 actions = 12 combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
@pytest.mark.parametrize("action", ACTIONS_TO_TEST, ids=[
    "perfect", "empty", "partial", "wrong_ids"
])
def test_score_strictly_between_zero_and_one(task: str, action: RegComplianceAction) -> None:
    """
    Mirrors EXACT Phase 2 validator check:
    'Each task score must be strictly between 0 and 1 (not 0.0 and not 1.0)'
    """
    env.reset(task=task)
    result = env.step(action)
    reward = result.reward

    assert reward > 0.0, (
        f"FAIL Phase 2: task={task} reward={reward!r} is exactly 0.0 "
        f"— judges reject this. violations={action.violation_ids}"
    )
    assert reward < 1.0, (
        f"FAIL Phase 2: task={task} reward={reward!r} is exactly 1.0 "
        f"— judges reject this. violations={action.violation_ids}"
    )


# ---------------------------------------------------------------------------
# Comprehensive safe_score() boundary tests
# ---------------------------------------------------------------------------

class TestSafeScore:
    """safe_score must NEVER return exactly 0.0 or 1.0."""

    def test_never_returns_exact_zero(self) -> None:
        assert safe_score(0.0) != 0.0
        assert safe_score(-0.0) != 0.0
        assert safe_score(-999.0) != 0.0
        assert safe_score(-0.0001) != 0.0

    def test_never_returns_exact_one(self) -> None:
        assert safe_score(1.0) != 1.0
        assert safe_score(1.0001) != 1.0
        assert safe_score(999.0) != 1.0
        assert safe_score(1 + 1e-10) != 1.0

    def test_boundary_float_precision(self) -> None:
        """Float precision edge cases that could slip through rounding."""
        tricky = [0.9999999, 0.99999, 0.000001, 0.9999950001]
        for v in tricky:
            result = safe_score(v)
            assert 0.0 < result < 1.0, f"safe_score({v}) = {result}"

    def test_infinity_and_nan(self) -> None:
        assert safe_score(float("inf")) == 0.95
        assert safe_score(-float("inf")) == 0.05
        # NaN: math.isfinite(NaN) is False, NaN > 0 is False → returns 0.05
        nan_result = safe_score(float("nan"))
        assert 0.0 < nan_result < 1.0, f"safe_score(NaN) = {nan_result}"
        assert nan_result == 0.05  # NaN > 0 is False, so goes to else branch → 0.05


    @pytest.mark.parametrize("raw,expected", [
        (0.0,    0.05),
        (-1.0,   0.05),
        (-100,   0.05),
        (1.0,    0.95),
        (2.0,    0.95),
        (100,    0.95),
        (0.5,    0.5),
        (0.95,   0.95),
        (0.05,   0.05),
        (0.3333, 0.3333),
        (0.7777, 0.7777),
    ])
    def test_boundary_values(self, raw: float, expected: float) -> None:
        assert safe_score(raw) == expected

    def test_strictly_less_than_one_for_all_inputs(self) -> None:
        for v in [0.0, 0.5, 1.0, -1.0, 2.0, 100.0, -100.0]:
            assert safe_score(v) < 1.0, f"safe_score({v}) >= 1.0"

    def test_strictly_greater_than_zero_for_all_inputs(self) -> None:
        for v in [0.0, 0.5, 1.0, -1.0, 2.0, 100.0, -100.0]:
            assert safe_score(v) > 0.0, f"safe_score({v}) <= 0.0"

    def test_double_application_is_idempotent(self) -> None:
        """Applying safe_score twice should give the same result."""
        for v in [0.0, 0.05, 0.3, 0.5, 0.7, 0.95, 1.0]:
            once = safe_score(v)
            twice = safe_score(once)
            assert once == twice, f"safe_score not idempotent at v={v}: {once} vs {twice}"


# ---------------------------------------------------------------------------
# All 12 task × action combinations in one test
# ---------------------------------------------------------------------------

def test_all_12_combinations() -> None:
    """
    Run all 3 tasks × 4 actions = 12 combinations.
    All must pass Phase 2 strict score check.
    """
    failures: list[str] = []
    for task in ("easy", "medium", "hard"):
        for action in ACTIONS_TO_TEST:
            env.reset(task=task)
            result = env.step(action)
            r = result.reward
            if not (0.0 < r < 1.0):
                failures.append(
                    f"task={task} violations={action.violation_ids} reward={r!r}"
                )

    assert not failures, (
        f"These {len(failures)} combinations FAIL Phase 2:\n"
        + "\n".join(failures)
    )


# ---------------------------------------------------------------------------
# Grade-specific boundary proof tests
# ---------------------------------------------------------------------------

class TestGradeEasyBoundaries:
    """Prove mathematically that easy grader cannot return 0.0 or 1.0."""

    def test_empty_action_above_zero(self) -> None:
        env.reset(task="easy")
        result = env.step(RegComplianceAction())
        assert result.reward > 0.0, f"Empty action → {result.reward}"
        assert result.reward < 1.0

    def test_perfect_action_below_one(self) -> None:
        env.reset(task="easy")
        result = env.step(RegComplianceAction(
            violation_ids=["ART6-CONSENT"],
            severity="high",
            explanation="Lacks lawful basis under Article 6; no explicit consent obtained.",
        ))
        assert result.reward < 1.0, f"Perfect easy action → {result.reward}"
        assert result.reward > 0.0


class TestGradeMediumBoundaries:
    def test_empty_action_above_zero(self) -> None:
        env.reset(task="medium")
        result = env.step(RegComplianceAction())
        assert result.reward > 0.0, f"Empty medium action → {result.reward}"
        assert result.reward < 1.0

    def test_all_concepts_matched_below_one(self) -> None:
        env.reset(task="medium")
        result = env.step(RegComplianceAction(
            violation_ids=["ART5-PURPOSE", "ART5-RETENTION", "ART6-CONSENT",
                           "ART13-TRANSPARENCY", "ART17-ERASURE"],
            severity="high",
            explanation="Multiple GDPR violations including purpose limitation, retention, consent, and transparency.",
        ))
        assert result.reward < 1.0, f"All-matched medium → {result.reward}"
        assert result.reward > 0.0


class TestGradeHardBoundaries:
    def test_empty_action_above_zero(self) -> None:
        env.reset(task="hard")
        result = env.step(RegComplianceAction())
        assert result.reward > 0.0, f"Empty hard action → {result.reward}"
        assert result.reward < 1.0

    def test_rich_action_below_one(self) -> None:
        env.reset(task="hard")
        long_fix = "Add explicit consent mechanism, define retention periods for all data categories, document lawful basis for processing, and implement erasure request workflow within 30 days."
        long_exp = "Version 1 lacked lawful basis, retention limits, and transparency. Version 2 fixed consent but still lacks erasure mechanism and full transparency on data recipients."
        result = env.step(RegComplianceAction(
            violation_ids=["ART6-LAWFUL-BASIS", "ART5-RETENTION", "ART13-TRANSPARENCY"],
            severity="high",
            explanation=long_exp,
            fix_suggestion=long_fix,
        ))
        assert result.reward < 1.0, f"Rich hard action → {result.reward}"
        assert result.reward > 0.0
