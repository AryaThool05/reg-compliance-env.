"""
inference.py — RegComplianceEnv inference script.

Runs the GDPR compliance checker against an LLM for all 3 task tiers
(easy, medium, hard) and prints machine-parseable results to stdout.

Required stdout format (machine-parsed by judges — zero tolerance):
  [START] task=<name> env=reg-compliance-env model=<MODEL_NAME>
  [STEP] step=<n> action=<str> reward=<0.00> done=<true|false> error=<null|msg>
  [END] success=<true|false> steps=<n> rewards=<r1,r2,...>

Environment variables:
  API_BASE_URL  — default: https://api.openai.com/v1   (MUST have default)
  MODEL_NAME    — default: gpt-4.1-mini                (MUST have default)
  HF_TOKEN      — NO default, raises ValueError if missing
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from openai import OpenAI

from models import RegComplianceObservation, RegComplianceAction
from task_definitions import TASK_CONFIGS, load_gdpr_articles, safe_score

# ---------------------------------------------------------------------------
# Configuration — EXACT variable names and defaults required by spec
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError(
        "HF_TOKEN environment variable is required but not set. "
        "Set it with: export HF_TOKEN=hf_..."
    )

# OpenAI client configured for HuggingFace or OpenAI router
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# Runtime constants
# ---------------------------------------------------------------------------

BENCHMARK_NAME: str = "reg-compliance-env"
TASK_IDS: list[str] = ["easy", "medium", "hard"]
MAX_TOKENS: int = 400          # stays within 8GB RAM / 2vCPU / 20min runtime
TEMPERATURE: float = 0.2
RATE_LIMIT_SLEEP: int = 13     # seconds — stays within 5 req/min
SUCCESS_SCORE_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# System prompt (per spec — instructs LLM to return raw JSON only)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a GDPR compliance expert auditing privacy policies.
Always respond with ONLY a JSON object in this exact format:
{"violation_ids": ["ART6-CONSENT"], "severity": "high", "explanation": "Brief explanation under 200 chars", "fix_suggestion": "Concrete fix suggestion under 200 chars"}
Use violation IDs like ART5-PURPOSE, ART6-CONSENT, ART6-LAWFUL-BASIS, ART13-TRANSPARENCY, ART17-ERASURE, ART5-RETENTION, ART5-MINIMISATION.
If no violations found, use violation_ids: [] and severity: none.
Do not use markdown code blocks. Output only the raw JSON object."""

# ---------------------------------------------------------------------------
# Structured stdout logging — EXACT format, machine-parsed by judges
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    """Print the [START] line. ONE per task."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    """Print the [STEP] line after each environment step."""
    done_str = "true" if done else "false"
    error_str = "null" if error is None else error
    # action must be a clean one-liner — no newlines, no quotes that break format
    clean_action = action.replace("\n", " ").replace('"', "'")[:120]
    print(
        f"[STEP] step={step} action={clean_action} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    """Print the [END] line. ONE per task. Always emitted (in finally block)."""
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def call_llm(observation: RegComplianceObservation) -> RegComplianceAction:
    """Call the LLM and parse its response into a RegComplianceAction.

    Sleeps RATE_LIMIT_SLEEP seconds after every API call.
    Falls back to a safe default Action if parsing fails.
    """
    user_prompt = observation.to_prompt()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        raw = (response.choices[0].message.content or "").strip()

        # Strip markdown fences if the model disobeys instructions
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        data = json.loads(raw)

        action = RegComplianceAction(
            violation_ids=data.get("violation_ids", []),
            severity=data.get("severity", "none"),
            explanation=str(data.get("explanation", ""))[:200],
            fix_suggestion=str(data.get("fix_suggestion", ""))[:200],
        )

    except Exception as exc:
        # Safe fallback — never let a parse error crash the run
        action = RegComplianceAction(
            violation_ids=[],
            severity="none",
            explanation=f"parse-error: {str(exc)[:80]}",
            fix_suggestion="",
        )

    # Rate limiting — 5 req/min = 12s between calls; 13s is safe
    time.sleep(RATE_LIMIT_SLEEP)

    return action


# ---------------------------------------------------------------------------
# Action summary for [STEP] log line
# ---------------------------------------------------------------------------

def _summarise_action(action: RegComplianceAction) -> str:
    """Create a short one-liner summary for the [STEP] action field."""
    if not action.violation_ids:
        return "no-violations-found"
    ids = "+".join(action.violation_ids[:4])
    count = len(action.violation_ids)
    suffix = f"+{count - 4}more" if count > 4 else ""
    return f"found:{ids}{suffix}"


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_id: str, gdpr_cache: dict[str, Any]) -> dict[str, Any]:
    """Run a single task end-to-end and emit structured log lines.

    Returns summary dict: {task, score, success}.
    """
    config = TASK_CONFIGS[task_id]
    rewards: list[float] = []
    steps = 0
    reward = 0.05
    success = False
    error_msg: str | None = None

    try:
        # ONE [START] per task
        log_start(task=task_id, env=BENCHMARK_NAME, model=MODEL_NAME)

        # Build observation (no network calls)
        observation = config["build_observation"](gdpr_cache)

        # Call LLM (rate-limited inside call_llm)
        action = call_llm(observation)

        # Grade
        ground_truth = config["ground_truth"]()
        reward = config["grader"](action, ground_truth)
        steps = 1
        rewards.append(reward)

        done = True
        success = reward >= SUCCESS_SCORE_THRESHOLD

        action_summary = _summarise_action(action)
        log_step(
            step=steps,
            action=action_summary,
            reward=reward,
            done=done,
            error=None,
        )

    except Exception as exc:
        error_msg = str(exc)[:120].replace("\n", " ")
        if steps == 0:
            steps = 1
            rewards.append(0.05)  # safe_score floor, not 0.0
            log_step(step=1, action="error-before-step", reward=0.05, done=True, error=error_msg)

    finally:
        # ONE [END] per task — ALWAYS emitted even on exception
        log_end(success=success, steps=steps, rewards=rewards)

    return {"task": task_id, "score": reward, "success": success}


# ---------------------------------------------------------------------------
# Main — runs ALL 3 tasks sequentially
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all task tiers sequentially and print a final summary."""
    # Load GDPR data once (static JSON, no network)
    gdpr_cache = load_gdpr_articles()

    results: list[dict[str, Any]] = []

    for task_id in TASK_IDS:
        result = run_task(task_id, gdpr_cache)
        results.append(result)

    # Human-readable summary (after all [END] lines)
    print("", flush=True)
    print("=" * 52, flush=True)
    print("INFERENCE SUMMARY", flush=True)
    print("=" * 52, flush=True)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  {r['task']:>8s}  {status}  score={r['score']:.2f}", flush=True)
    avg = sum(r["score"] for r in results) / len(results) if results else 0.05
    passed = sum(1 for r in results if r["success"])
    print(f"  {'avg':>8s}       score={avg:.2f}  ({passed}/{len(results)} passed)", flush=True)
    print("=" * 52, flush=True)


if __name__ == "__main__":
    main()
