"""
RegComplianceEnv inference script.

Runs the GDPR compliance checker against an LLM for all task tiers
(easy, medium, hard) and prints machine-parseable results to stdout.

The [START] / [STEP] / [END] format is MANDATORY — judges parse it exactly.
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from environment import RegComplianceEnv
from models import Action, Observation

# ---------------------------------------------------------------------------
# Load .env (no-op if missing) and read configuration
# ---------------------------------------------------------------------------

load_dotenv()

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")  # NO default — None if not set
API_KEY: str = HF_TOKEN or os.getenv("API_KEY", "")  # fallback to API_KEY only
FIRECRAWL_API_KEY: str = os.getenv("FIRECRAWL_API_KEY", "")

MAX_STEPS: int = 1
TEMPERATURE: float = 0.3
MAX_TOKENS: int = 400
SUCCESS_SCORE_THRESHOLD: float = 0.5
RATE_LIMIT_SLEEP: int = 13  # seconds — stays within 5 req/min
TASK_IDS: list[str] = ["easy", "medium", "hard"]
BENCHMARK_NAME: str = "reg-compliance-env"


# ---------------------------------------------------------------------------
# Structured stdout logging (machine-parsed by judges)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    """Print the [START] line for a task."""
    print(f"[START] task={task} env={env} model={model}")


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    """Print the [STEP] line after an environment step."""
    done_str = "true" if done else "false"
    error_str = "null" if error is None else error
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}")


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    """Print the [END] line summarising the task."""
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rewards_str}")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    """Return a concise system prompt for the compliance-checking LLM."""
    return textwrap.dedent("""\
        You are a GDPR compliance auditor. You will receive a regulation text
        and a company privacy policy clause. Analyse the policy for GDPR
        violations and respond with a JSON object containing exactly these keys:

        {
          "violation_ids": ["ART<N>-<LABEL>", ...],
          "severity": "none" | "low" | "medium" | "high",
          "explanation": "...",
          "fix_suggestion": "..." or null
        }

        Rules:
        - violation_ids: list of strings like "ART6-CONSENT", "ART5-RETENTION".
          Use the format ART<article_number>-<SHORT_LABEL>.
          Return an empty list [] if no violations found.
        - severity: overall severity of the worst violation.
        - explanation: brief explanation of each violation found.
        - fix_suggestion: concrete recommendation to fix the issues, or null.

        Respond ONLY with valid JSON. No markdown fences, no commentary.""")


def build_user_prompt(observation: Observation) -> str:
    """Return the user prompt from the observation's to_prompt() method."""
    return observation.to_prompt()


# ---------------------------------------------------------------------------
# Action summary for log output
# ---------------------------------------------------------------------------

def _summarise_action(action: Action) -> str:
    """Create a short human-readable summary of an action for [STEP] output."""
    if not action.violation_ids:
        return "No violations found"
    ids = ", ".join(action.violation_ids[:5])
    count = len(action.violation_ids)
    return f"Found {count} violation{'s' if count != 1 else ''}: {ids}"


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

async def get_model_action(client: OpenAI, observation: Observation) -> Action:
    """Call the LLM and parse its response into an Action.

    Falls back to a safe default Action if the response cannot be parsed.
    Sleeps ``RATE_LIMIT_SLEEP`` seconds after each API call.
    """
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(observation)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        raw = response.choices[0].message.content or ""

        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            # Remove first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        data = json.loads(raw)

        action = Action(
            violation_ids=data.get("violation_ids", []),
            severity=data.get("severity", "none"),
            explanation=data.get("explanation", ""),
            fix_suggestion=data.get("fix_suggestion"),
        )

    except Exception as exc:
        # Safe fallback so the run never crashes
        action = Action(
            violation_ids=[],
            severity="none",
            explanation=f"parse error: {exc}",
            fix_suggestion=None,
        )

    # Rate limit: sleep to stay under 5 req/min
    time.sleep(RATE_LIMIT_SLEEP)

    return action


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, env: RegComplianceEnv, task_id: str) -> dict[str, Any]:
    """Run a single task tier end-to-end and print structured logs.

    Returns a summary dict with task, score, and success.
    """
    rewards: list[float] = []
    steps = 0
    score = 0.0
    success = False
    error_msg: str | None = None

    try:
        log_start(task=task_id, env=BENCHMARK_NAME, model=MODEL_NAME)

        # Reset environment
        reset_result = await env.reset(task=task_id)
        observation = reset_result.observation

        # Get model action
        action = await get_model_action(client, observation)

        # Step environment
        step_result = await env.step(action)
        steps += 1
        rewards.append(step_result.reward)

        action_summary = _summarise_action(action)
        log_step(
            step=steps,
            action=action_summary,
            reward=step_result.reward,
            done=step_result.done,
            error=None,
        )

        score = step_result.reward
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        error_msg = str(exc)
        if steps == 0:
            # Error happened before any step was logged
            log_step(step=1, action="error", reward=0.0, done=True, error=error_msg)
            steps = 1
            rewards.append(0.0)

    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)

    return {"task": task_id, "score": score, "success": success}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run all task tiers sequentially and print a final summary."""
    if not API_KEY:
        raise RuntimeError(
            "No API key found. Set HF_TOKEN or API_KEY in your environment."
        )

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    env = RegComplianceEnv()
    results: list[dict[str, Any]] = []

    for task_id in TASK_IDS:
        result = await run_task(client, env, task_id)
        results.append(result)

    await env.close()

    # ---- Summary ----
    total_score = sum(r["score"] for r in results)
    avg_score = total_score / len(results) if results else 0.0
    passed = sum(1 for r in results if r["success"])

    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  {r['task']:>8s}  {status}  score={r['score']:.3f}")
    print(f"  {'':>8s}  ---")
    print(f"  {'avg':>8s}       score={avg_score:.3f}  ({passed}/{len(results)} passed)")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
