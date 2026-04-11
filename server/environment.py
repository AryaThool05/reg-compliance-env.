"""
server/environment.py — RegComplianceEnvironment

Standalone implementation with NO openenv.core imports.
Bypasses create_app entirely to avoid hidden framework requirements.

Return type contract:
- reset()  → RegComplianceObservation (Pydantic model, NEVER dict)
- step()   → StepResult (.observation, .reward, .done, .info)
- state    → RegComplianceState (property, Pydantic model)
"""

from __future__ import annotations

import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import RegComplianceObservation, RegComplianceAction, RegComplianceState

# ---------------------------------------------------------------------------
# Static GDPR fallback — never raises, never reads files
# ---------------------------------------------------------------------------

STATIC_GDPR = {
    "5": {
        "title": "Article 5 — Principles relating to processing",
        "text": (
            "Personal data shall be: processed lawfully, fairly and transparently; "
            "collected for specified, explicit and legitimate purposes (purpose limitation); "
            "adequate, relevant and limited to what is necessary (data minimisation); "
            "accurate and kept up to date; kept no longer than necessary (storage limitation); "
            "processed in a manner that ensures appropriate security (integrity and confidentiality)."
        ),
        "full_text": (
            "Personal data shall be: processed lawfully, fairly and transparently; "
            "collected for specified, explicit and legitimate purposes (purpose limitation); "
            "adequate, relevant and limited to what is necessary (data minimisation); "
            "accurate and kept up to date; kept no longer than necessary (storage limitation); "
            "processed in a manner that ensures appropriate security (integrity and confidentiality)."
        ),
    },
    "6": {
        "title": "Article 6 — Lawfulness of processing",
        "text": (
            "Processing shall be lawful only if the data subject has given consent, or processing "
            "is necessary for the performance of a contract, or compliance with a legal obligation, "
            "or protection of vital interests, or performance of a task in the public interest, "
            "or for the purposes of the legitimate interests pursued by the controller."
        ),
        "full_text": (
            "Processing shall be lawful only if the data subject has given consent, or processing "
            "is necessary for the performance of a contract, or compliance with a legal obligation, "
            "or protection of vital interests, or performance of a task in the public interest, "
            "or for the purposes of the legitimate interests pursued by the controller."
        ),
    },
    "13": {
        "title": "Article 13 — Information to be provided",
        "text": (
            "Where personal data are collected from the data subject, the controller shall provide: "
            "the identity and contact details of the controller; the purposes and legal basis for "
            "processing; the recipients of data; the period for which data will be stored; "
            "and the data subject's rights including access, rectification, erasure, and complaint."
        ),
        "full_text": (
            "Where personal data are collected from the data subject, the controller shall provide: "
            "the identity and contact details of the controller; the purposes and legal basis for "
            "processing; the recipients of data; the period for which data will be stored; "
            "and the data subject's rights including access, rectification, erasure, and complaint."
        ),
    },
    "17": {
        "title": "Article 17 — Right to erasure",
        "text": (
            "The data subject shall have the right to obtain erasure of personal data without "
            "undue delay where the data is no longer necessary, consent is withdrawn, "
            "the data has been unlawfully processed, or erasure is required by law."
        ),
        "full_text": (
            "The data subject shall have the right to obtain erasure of personal data without "
            "undue delay where the data is no longer necessary, consent is withdrawn, "
            "the data has been unlawfully processed, or erasure is required by law."
        ),
    },
}

SAMPLE_POLICIES = {
    "easy": (
        "We collect your email address and share it with our marketing partners. "
        "We use your data as we see fit without requiring your explicit consent. "
        "By using our service you agree to all data sharing."
    ),
    "medium": (
        "Our platform collects personal data including name, email, location, browsing history, "
        "and purchase history. We share this data with third-party advertisers for targeting. "
        "Data is kept indefinitely. Users cannot request deletion of their data. "
        "We may use data for purposes not originally stated at collection time. "
        "No retention period is specified anywhere in this policy."
    ),
    "hard_v1": (
        "We collect user data and share with partners. No retention period specified. "
        "Users cannot delete their data. We use data for undisclosed purposes. "
        "No lawful basis stated for any processing activity."
    ),
    "hard_v2": (
        "We collect user data with your consent for specified purposes only. "
        "Data is retained for 2 years then deleted. Users may request deletion by contacting us. "
        "We share data only with essential service providers under data processing agreements. "
        "Our lawful basis for processing is consent and contractual necessity."
    ),
}


# ---------------------------------------------------------------------------
# StepResult — standalone, no openenv.core dependency
# ---------------------------------------------------------------------------

class StepResult:
    """Return type for step(). Holds observation, reward, done, info."""

    def __init__(
        self,
        observation: RegComplianceObservation,
        reward: float,
        done: bool,
        info: dict,
    ) -> None:
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info


# ---------------------------------------------------------------------------
# Grading utility
# ---------------------------------------------------------------------------

def safe_score(s: float) -> float:
    """Score MUST be strictly between 0.0 and 1.0 exclusive (0.05 floor, 0.95 ceiling)."""
    try:
        s = float(s)
    except Exception:
        return 0.05
    if s <= 0.0:
        return 0.05
    if s >= 1.0:
        return 0.95
    return round(s, 4)


# ---------------------------------------------------------------------------
# Main environment class — NO openenv.core dependency
# ---------------------------------------------------------------------------

class RegComplianceEnvironment:
    """Standalone GDPR compliance environment.

    No openenv.core imports — pure Python class that implements the
    required interface manually.
    """

    def __init__(self) -> None:
        self._current_task: str = "easy"
        self._step_count: int = 0
        self._done: bool = False
        self._episode_id: str = str(uuid.uuid4())
        self._gdpr: dict = self._load_gdpr()

    # ---- GDPR data loading ------------------------------------------------

    def _load_gdpr(self) -> dict:
        """Load GDPR articles from static JSON. Falls back to hardcoded data."""
        try:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base, "data", "gdpr_articles.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return STATIC_GDPR

    def _article_text(self, key: str) -> str:
        """Extract article text from the loaded GDPR data safely."""
        entry = self._gdpr.get(key, STATIC_GDPR.get(key, {}))
        if isinstance(entry, dict):
            return entry.get("full_text", entry.get("text", entry.get("summary", "")))
        return str(entry)

    # ---- Observation builders ---------------------------------------------

    def _build_observation(self, task: str) -> RegComplianceObservation:
        """Build a RegComplianceObservation for the given task. Never raises."""
        try:
            if task == "easy":
                return RegComplianceObservation(
                    regulation_text=self._article_text("6"),
                    policy_text=SAMPLE_POLICIES["easy"],
                    task_id="easy",
                    article_refs=["Article 6"],
                    instructions=(
                        "Check if this policy clause violates GDPR Article 6 (lawful basis "
                        "for processing). Use violation IDs like ART6-CONSENT, ART6-LAWFUL-BASIS."
                    ),
                    context={"difficulty": "easy"},
                )
            elif task == "medium":
                reg_text = " | ".join(
                    self._article_text(k) for k in ("5", "6", "13")
                )
                return RegComplianceObservation(
                    regulation_text=reg_text,
                    policy_text=SAMPLE_POLICIES["medium"],
                    task_id="medium",
                    article_refs=["Article 5", "Article 6", "Article 13"],
                    instructions=(
                        "Audit this full privacy policy against GDPR Articles 5, 6, and 13. "
                        "List ALL violations using IDs like ART5-PURPOSE, ART5-RETENTION, "
                        "ART6-LAWFUL-BASIS, ART13-TRANSPARENCY."
                    ),
                    context={"difficulty": "medium"},
                )
            else:  # hard
                reg_text = " | ".join(
                    self._article_text(k) for k in ("5", "6", "13", "17")
                )
                policy_text = (
                    f"POLICY VERSION 1 (original):\n{SAMPLE_POLICIES['hard_v1']}\n\n"
                    f"POLICY VERSION 2 (updated):\n{SAMPLE_POLICIES['hard_v2']}"
                )
                return RegComplianceObservation(
                    regulation_text=reg_text,
                    policy_text=policy_text,
                    task_id="hard",
                    article_refs=["Article 5", "Article 6", "Article 13", "Article 17"],
                    instructions=(
                        "Compare policy VERSION 1 and VERSION 2. List all GDPR violations found, "
                        "which were fixed between versions, and provide a fix_suggestion for "
                        "violations that remain in VERSION 2."
                    ),
                    context={"difficulty": "hard"},
                )
        except Exception as exc:
            return RegComplianceObservation(
                regulation_text=STATIC_GDPR["6"]["text"],
                policy_text=SAMPLE_POLICIES["easy"],
                task_id="easy",
                article_refs=["Article 6"],
                instructions="Identify GDPR violations in the policy text.",
                context={"error": str(exc)},
            )

    # ---- Public interface -------------------------------------------------

    def reset(self, task: str = "easy") -> RegComplianceObservation:
        """Reset environment for a new episode. Returns RegComplianceObservation."""
        try:
            if isinstance(task, dict):
                task = task.get("task", task.get("task_id", "easy"))
            if not isinstance(task, str) or task not in ("easy", "medium", "hard"):
                task = "easy"

            self._current_task = task
            self._step_count = 0
            self._done = False
            self._episode_id = str(uuid.uuid4())

            return self._build_observation(task)

        except Exception as exc:
            self._current_task = "easy"
            self._step_count = 0
            self._done = False
            self._episode_id = str(uuid.uuid4())
            return RegComplianceObservation(
                regulation_text=STATIC_GDPR["6"]["text"],
                policy_text=SAMPLE_POLICIES["easy"],
                task_id="easy",
                article_refs=["Article 6"],
                instructions="Identify GDPR violations in the policy text.",
                context={"error": str(exc)},
            )

    def step(self, action=None) -> StepResult:
        """Execute one step. Returns StepResult with Pydantic observation."""
        try:
            if action is None:
                action = RegComplianceAction()
            elif isinstance(action, dict):
                action = RegComplianceAction(**{
                    k: v for k, v in action.items()
                    if k in RegComplianceAction.model_fields
                })

            self._step_count += 1
            self._done = True

            score = self._grade(action)
            obs = self._build_observation(self._current_task)

            return StepResult(
                observation=obs,
                reward=safe_score(score),
                done=True,
                info={"task": self._current_task, "score": safe_score(score)},
            )

        except Exception as exc:
            self._step_count += 1
            self._done = True
            return StepResult(
                observation=self._build_observation(self._current_task),
                reward=0.05,
                done=True,
                info={"error": str(exc)[:200]},
            )

    def _grade(self, action: RegComplianceAction) -> float:
        """Score the action against the current task. Always returns safe_score result."""
        violations = [v.upper() for v in (action.violation_ids or [])]

        if self._current_task == "easy":
            found = any(
                kw in v for v in violations
                for kw in ("ART6", "CONSENT", "LAWFUL", "BASIS")
            )
            base = 0.45 if found else 0.05
            has_explanation = 0.45 if action.explanation and len(action.explanation) > 10 else 0.05
            return safe_score(base + has_explanation)

        elif self._current_task == "medium":
            concept_groups = [
                ("ART5", "PURPOSE", "MINIMIS", "RETENTION", "STORAGE"),
                ("ART6", "LAWFUL", "CONSENT", "BASIS"),
                ("ART13", "TRANSPARENT", "INFORM", "DISCLOS"),
                ("ART17", "ERASURE", "DELET", "FORGET"),
            ]
            matched = sum(
                1 for kws in concept_groups
                if any(any(kw in v for kw in kws) for v in violations)
            )
            total = len(concept_groups)
            recall = matched / total if total else 0.05
            valid = sum(
                1 for v in violations
                if any(any(kw in v for kw in kws) for kws in concept_groups)
            )
            precision = valid / len(violations) if violations else 0.05
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.05)
            return safe_score(f1 * 0.90 + 0.05)

        else:  # hard
            has_violations = 0.85 if len(violations) >= 2 else 0.25
            has_fix = 0.85 if action.fix_suggestion and len(action.fix_suggestion) > 50 else 0.20
            has_explanation = 0.85 if action.explanation and len(action.explanation) > 50 else 0.20
            raw = 0.40 * has_violations + 0.35 * has_fix + 0.25 * has_explanation
            return safe_score(raw)

    @property
    def state(self) -> RegComplianceState:
        """Current environment state as a Pydantic model."""
        return RegComplianceState(
            task_id=self._current_task,
            step_count=self._step_count,
            done=self._done,
            episode_id=self._episode_id,
        )

    def close(self) -> None:
        """No-op. Required by some base class interfaces."""
        pass

    async def reset_async(self, **kwargs):
        """Async wrapper for reset()."""
        task = kwargs.get("task", kwargs.get("task_id", "easy"))
        return self.reset(task=task)

    async def step_async(self, action=None, **kwargs):
        """Async wrapper for step()."""
        return self.step(action)
