"""
tests/test_inference_logging.py

Tests that the structured stdout log functions produce exactly the format
the judges parse. Zero tolerance on format deviations.

Expected formats:
  [START] task=<name> env=reg-compliance-env model=<MODEL_NAME>
  [STEP] step=<n> action=<str> reward=<0.00> done=<true|false> error=<null|msg>
  [END] success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

from __future__ import annotations

import re
import sys
import os
import io
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Temporarily set HF_TOKEN so import doesn't raise
os.environ.setdefault("HF_TOKEN", "test-token-for-unit-tests")

import importlib


def _import_log_functions():
    """Import log functions without triggering the HF_TOKEN guard at module scope."""
    # We need to import only the log functions, not trigger the ValueError
    # We'll test the functions in isolation by importing them with a mock token
    import inference
    return inference.log_start, inference.log_step, inference.log_end


class TestLogStart:
    def test_format(self, capsys) -> None:
        log_start, _, _ = _import_log_functions()
        log_start(task="easy", env="reg-compliance-env", model="gpt-4.1-mini")
        captured = capsys.readouterr()
        assert captured.out.strip() == "[START] task=easy env=reg-compliance-env model=gpt-4.1-mini"

    def test_contains_all_fields(self, capsys) -> None:
        log_start, _, _ = _import_log_functions()
        log_start(task="hard", env="reg-compliance-env", model="test-model")
        captured = capsys.readouterr()
        line = captured.out.strip()
        assert line.startswith("[START]")
        assert "task=hard" in line
        assert "env=reg-compliance-env" in line
        assert "model=test-model" in line


class TestLogStep:
    def test_format_no_error(self, capsys) -> None:
        _, log_step, _ = _import_log_functions()
        log_step(step=1, action="found:ART6-CONSENT", reward=0.95, done=True, error=None)
        captured = capsys.readouterr()
        line = captured.out.strip()
        assert line.startswith("[STEP]")
        assert "step=1" in line
        assert "reward=0.95" in line
        assert "done=true" in line
        assert "error=null" in line

    def test_done_false_format(self, capsys) -> None:
        _, log_step, _ = _import_log_functions()
        log_step(step=2, action="no-violations-found", reward=0.05, done=False, error=None)
        captured = capsys.readouterr()
        assert "done=false" in captured.out

    def test_reward_two_decimal_places(self, capsys) -> None:
        _, log_step, _ = _import_log_functions()
        log_step(step=1, action="test", reward=0.333333, done=True, error=None)
        captured = capsys.readouterr()
        assert "reward=0.33" in captured.out

    def test_error_field_non_null(self, capsys) -> None:
        _, log_step, _ = _import_log_functions()
        log_step(step=1, action="error", reward=0.05, done=True, error="parse failed")
        captured = capsys.readouterr()
        assert "error=parse failed" in captured.out

    def test_done_values_are_lowercase(self, capsys) -> None:
        _, log_step, _ = _import_log_functions()
        log_step(step=1, action="x", reward=0.5, done=True, error=None)
        out1 = capsys.readouterr().out
        log_step(step=1, action="x", reward=0.5, done=False, error=None)
        out2 = capsys.readouterr().out
        assert "true" in out1 and "True" not in out1
        assert "false" in out2 and "False" not in out2


class TestLogEnd:
    def test_format(self, capsys) -> None:
        _, _, log_end = _import_log_functions()
        log_end(success=True, steps=1, rewards=[0.87])
        captured = capsys.readouterr()
        line = captured.out.strip()
        assert line.startswith("[END]")
        assert "success=true" in line
        assert "steps=1" in line
        assert "rewards=0.87" in line

    def test_multiple_rewards(self, capsys) -> None:
        _, _, log_end = _import_log_functions()
        log_end(success=False, steps=3, rewards=[0.05, 0.50, 0.95])
        captured = capsys.readouterr()
        assert "rewards=0.05,0.50,0.95" in captured.out

    def test_success_false_format(self, capsys) -> None:
        _, _, log_end = _import_log_functions()
        log_end(success=False, steps=1, rewards=[0.05])
        captured = capsys.readouterr()
        assert "success=false" in captured.out

    def test_no_score_field(self, capsys) -> None:
        """[END] must NOT contain 'score=' — only 'rewards='."""
        _, _, log_end = _import_log_functions()
        log_end(success=True, steps=1, rewards=[0.75])
        captured = capsys.readouterr()
        assert "score=" not in captured.out

    def test_rewards_two_decimal_places(self, capsys) -> None:
        _, _, log_end = _import_log_functions()
        log_end(success=True, steps=2, rewards=[0.333, 0.667])
        captured = capsys.readouterr()
        assert "rewards=0.33,0.67" in captured.out
