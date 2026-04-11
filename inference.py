"""
inference.py — RegComplianceEnv inference script.

Runs the GDPR compliance checker against an LLM for all 3 task tiers
(easy, medium, hard) and prints machine-parseable results to stdout.

MANDATORY stdout format (machine-parsed by judges — zero tolerance):
  [START] task=<name> env=reg-compliance-env model=<MODEL_NAME>
  [STEP] step=<n> action=<str> reward=<0.00> done=<true|false> error=<null|msg>
  [END] success=<true|false> steps=<n> rewards=<r1,r2,...>

Rules:
  - reward / rewards: exactly 2 decimal places  (0.72 not 0.7 not 0.720)
  - done / success: lowercase true or false      (never True or False)
  - error: the word null                         (not None, not "None")
  - ONE [START] per task, ONE [STEP] per task, ONE [END] per task
  - [END] always emitted in finally block — even on exception
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

from openai import OpenAI

# ---------------------------------------------------------------------------
# Project root on sys.path so direct imports work when run from any CWD
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import RegComplianceObservation, RegComplianceAction
from server.environment import RegComplianceEnvironment

# ---------------------------------------------------------------------------
# Config — EXACT variable names and defaults required by spec
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required but not set.")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# Runtime constants
# ---------------------------------------------------------------------------

BENCHMARK_NAME: str = "reg-compliance-env"
TASK_IDS: list[str] = ["easy", "medium", "hard"]
MAX_TOKENS: int = 400
TEMPERATURE: float = 0.3
RATE_LIMIT_SLEEP: int = 13          # 5 req/min limit → 13 s gap is safe
SUCCESS_SCORE_THRESHOLD: float = 0.3

# ---------------------------------------------------------------------------
# System prompt — LLM must return ONLY raw JSON, no markdown
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a GDPR compliance expert auditing privacy policies.\n"
    "Always respond with ONLY a JSON object in this exact format:\n"
    '{"violation_ids": ["ART6-CONSENT"], "severity": "high", '
    '"explanation": "brief explanation", "fix_suggestion": "concrete fix"}\n'
    "Use violation IDs like: ART5-PURPOSE, ART5-RETENTION, ART5-MINIMISATION, "
    "ART6-CONSENT, ART6-LAWFUL-BASIS, ART13-TRANSPARENCY, ART17-ERASURE.\n"
    "If no violations found, use violation_ids: [] and severity: none.\n"
    "No markdown. No code blocks. Output only the raw JSON object."
)

# ---------------------------------------------------------------------------
# Structured stdout logging — EXACT format, machine-parsed by judges
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    """ONE [START] line per task."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: str | None,
) -> None:
    """ONE [STEP] line per task step."""
    done_str = "true" if done else "false"
    error_str = "null" if error is None else str(error).replace("\n", " ")[:120]
    # Action: single line, no newlines, no quotes, max 100 chars
    clean_action = (
        str(action)
        .replace("\n", " ")
        .replace("\r", "")
        .replace('"', "")
        .replace("'", "")
        .strip()[:100]
    )
    print(
        f"[STEP] step={step} action={clean_action} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    """ONE [END] line per task — ALWAYS emitted in finally block."""
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Action string for [STEP] log line
# ---------------------------------------------------------------------------


def _action_summary(action: RegComplianceAction) -> str:
    """Format: 'violations=ART6-CONSENT,ART5-RETENTION severity=high'.

    Max 100 chars. Single line. No special chars that break log parsing.
    """
    if not action.violation_ids:
        return f"violations=none severity={action.severity}"
    ids = ",".join(action.violation_ids[:4])
    summary = f"violations={ids} severity={action.severity}"
    return summary[:100]


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


async def get_model_action(obs: RegComplianceObservation) -> RegComplianceAction:
    """Call the LLM and parse the response into a RegComplianceAction.

    NEVER raises — falls back to RegComplianceAction() with defaults on any error.
    Sleeps RATE_LIMIT_SLEEP seconds after every API call.
    """
    action = RegComplianceAction()  # safe default

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs.to_prompt()},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        raw = (response.choices[0].message.content or "").strip()

        # Bulletproof markdown stripping
        # Step 1: strip surrounding backtick fences
        raw = raw.strip("`")
        # Step 2: strip language tag if present (```json\n...)
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        # Step 3: remove any remaining ``` lines
        lines = [ln for ln in raw.splitlines() if ln.strip() not in ("```", "~~~")]
        raw = "\n".join(lines).strip()

        # Parse JSON
        data: dict[str, Any] = json.loads(raw)

        action = RegComplianceAction(
            violation_ids=[
                str(v) for v in data.get("violation_ids", [])
                if isinstance(v, str)
            ],
            severity=str(data.get("severity", "none")),
            explanation=str(data.get("explanation", ""))[:200],
            fix_suggestion=str(data.get("fix_suggestion", ""))[:200],
        )

    except Exception:
        # Any failure → return default action, never propagate
        action = RegComplianceAction()

    finally:
        # Rate limit: 5 req/min = sleep 13 s after every call
        time.sleep(RATE_LIMIT_SLEEP)

    return action


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------


async def run_task(env: RegComplianceEnvironment, task_id: str) -> dict[str, Any]:
    """Run one task end-to-end and emit [START] / [STEP] / [END] log lines.

    Structure:
      1. log_start
      2. env.reset(task_id)   → observation
      3. get_model_action(obs) → action  (LLM call + rate-limit sleep)
      4. env.step(action)      → result with reward
      5. log_step
      6. log_end (in finally — always emitted)

    Returns: {"task": str, "reward": float, "success": bool}
    """
    rewards: list[float] = []
    steps: int = 0
    reward: float = 0.05
    success: bool = False
    error_msg: str | None = None

    try:
        # ── [START] ────────────────────────────────────────────────────────
        log_start(task=task_id, env=BENCHMARK_NAME, model=MODEL_NAME)

        # ── reset ─────────────────────────────────────────────────────────
        obs: RegComplianceObservation = env.reset(task=task_id)

        # ── LLM call ──────────────────────────────────────────────────────
        action: RegComplianceAction = await get_model_action(obs)

        # ── step ──────────────────────────────────────────────────────────
        step_result = env.step(action)
        reward = float(getattr(step_result, "reward", 0.05))
        steps = 1
        rewards.append(reward)

        success = reward >= SUCCESS_SCORE_THRESHOLD

        # ── [STEP] ────────────────────────────────────────────────────────
        log_step(
            step=steps,
            action=_action_summary(action),
            reward=reward,
            done=True,
            error=None,
        )

    except Exception as exc:
        error_msg = str(exc).replace("\n", " ")[:120]
        if steps == 0:
            steps = 1
            reward = 0.05
            rewards.append(reward)
            log_step(
                step=1,
                action="parse_error",
                reward=0.05,
                done=True,
                error=error_msg,
            )

    finally:
        # ── [END] — ALWAYS emitted ─────────────────────────────────────────
        log_end(success=success, steps=steps, rewards=rewards)

    return {"task": task_id, "reward": reward, "success": success}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run all 3 tasks sequentially. Shares a single env instance."""
    env = RegComplianceEnvironment()
    results: list[dict[str, Any]] = []

    for task_id in TASK_IDS:
        result = await run_task(env, task_id)
        results.append(result)

    # ── Human-readable summary (after all machine-parsed lines) ──────────
    total_reward = sum(r["reward"] for r in results)
    avg_reward = total_reward / len(results) if results else 0.05
    passed = sum(1 for r in results if r["success"])

    print("", flush=True)
    print("=" * 52, flush=True)
    print("INFERENCE SUMMARY", flush=True)
    print("=" * 52, flush=True)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  {r['task']:>8s}  {status}  reward={r['reward']:.2f}", flush=True)
    print(
        f"  {'avg':>8s}       reward={avg_reward:.2f}  "
        f"({passed}/{len(results)} passed)",
        flush=True,
    )
    print("=" * 52, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
