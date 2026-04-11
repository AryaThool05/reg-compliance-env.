"""
task_definitions.py — all 3 task configurations, ground truths, and graders.

This is the single source of truth for task logic. Every grader uses
safe_score() to ensure rewards are strictly in (0.05, 0.95).

RULE: 0.0 is INVALID. 1.0 is INVALID. Only (0.05 to 0.95) is valid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from models import RegComplianceObservation, RegComplianceAction
except ImportError:
    from .models import RegComplianceObservation, RegComplianceAction

_PROJECT_ROOT = Path(__file__).resolve().parent
_DATA_DIR = _PROJECT_ROOT / "data"
_POLICIES_DIR = _DATA_DIR / "sample_policies"


# ---------------------------------------------------------------------------
# safe_score — MANDATORY for all graders (never returns 0.0 or 1.0)
# ---------------------------------------------------------------------------

def safe_score(raw: float) -> float:
    """Clamp a raw score to the strictly-exclusive range (0.05, 0.95).

    NEVER returns 0.0 or 1.0. Never use max(0, min(1, x)) in graders.
    - Perfect match → 0.95 (not 1.0)
    - No match      → 0.05 (not 0.0)
    - Partial       → interpolated between 0.05 and 0.95
    """
    if raw <= 0.0:
        return 0.05
    if raw >= 1.0:
        return 0.95
    return round(min(0.95, max(0.05, raw)), 4)


# ---------------------------------------------------------------------------
# GDPR article loader (static, no network calls)
# ---------------------------------------------------------------------------

def load_gdpr_articles() -> dict[str, Any]:
    """Load GDPR article summaries from the static JSON file.

    Returns dict keyed by article number string ("5", "6", etc.).
    Falls back to minimal inline data if file is missing.
    """
    json_path = _DATA_DIR / "gdpr_articles.json"
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)

    # Minimal inline fallback (should never be needed in Docker)
    return {
        "5": {"title": "Art 5", "summary": "Data minimisation and purpose limitation.", "full_text": "Personal data shall be processed lawfully, fairly and transparently. Purpose limitation, data minimisation, accuracy, storage limitation, integrity and confidentiality.", "key_violations": ["ART5-PURPOSE", "ART5-RETENTION", "ART5-MINIMISATION"]},
        "6": {"title": "Art 6", "summary": "Lawfulness of processing.", "full_text": "Processing shall be lawful only if: consent given, contract necessity, legal obligation, vital interests, public task, or legitimate interests.", "key_violations": ["ART6-CONSENT", "ART6-LAWFUL-BASIS"]},
        "13": {"title": "Art 13", "summary": "Transparency at collection.", "full_text": "Controller must provide: identity, purposes, legal basis, recipients, retention period, and data subject rights.", "key_violations": ["ART13-TRANSPARENCY"]},
    }


def _read_policy(filename: str) -> str:
    """Read a sample policy file. Raises FileNotFoundError if missing."""
    path = _POLICIES_DIR / filename
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# TASK 1 — easy
# ---------------------------------------------------------------------------

def _build_easy_observation(gdpr: dict[str, Any]) -> RegComplianceObservation:
    art6 = gdpr.get("6", {})
    regulation_text = art6.get("full_text", art6.get("summary", "")) if isinstance(art6, dict) else str(art6)

    return RegComplianceObservation(
        regulation_text=regulation_text,
        policy_text=(
            "We share your personal data with marketing partners without "
            "requiring your explicit consent. Users cannot opt out of this sharing."
        ),
        task_id="easy",
        article_refs=["Article 6"],
        instructions=(
            "Identify GDPR violations in the policy clause above. "
            "Check whether the policy has a valid lawful basis for data processing under Article 6. "
            "Return violation IDs like ART6-CONSENT or ART6-LAWFUL-BASIS if violations exist."
        ),
        context={"source": "synthetic", "difficulty": "easy"},
    )


def _ground_truth_easy() -> dict[str, Any]:
    return {
        "violation_ids": ["ART6-CONSENT"],
        "severity": "high",
        "article": "6",
    }


def grade_easy(action: RegComplianceAction, ground_truth: dict[str, Any]) -> float:
    """Grade easy task: keyword-based flexible matching against Article 6 concepts.

    Formula:
        score = safe_score(0.45 * violation_found + 0.45 * article_correct + 0.05)

    This always gives a range within [0.05, 0.95] — strictly exclusive.
    """
    CONCEPT_KEYWORDS = {"ART6", "CONSENT", "LAWFUL", "BASIS", "GDPR"}

    predicted_upper = [v.upper() for v in action.violation_ids]

    # violation_found: did agent flag ANY violation?
    violation_found = 1.0 if action.violation_ids else 0.0

    # article_correct: does any violation ID contain concepts related to Article 6?
    article_correct = 0.0
    if predicted_upper:
        for vid in predicted_upper:
            if any(kw in vid for kw in CONCEPT_KEYWORDS):
                article_correct = 1.0
                break

    raw = 0.45 * violation_found + 0.45 * article_correct + 0.05
    return safe_score(raw)


# ---------------------------------------------------------------------------
# TASK 2 — medium
# ---------------------------------------------------------------------------

def _build_medium_observation(gdpr: dict[str, Any]) -> RegComplianceObservation:
    parts: list[str] = []
    for art_num in ("5", "6", "13"):
        entry = gdpr.get(art_num, {})
        text = entry.get("full_text", "") if isinstance(entry, dict) else str(entry)
        if text:
            parts.append(f"=== Article {art_num} ===\n{text}")
    regulation_text = "\n\n".join(parts)

    policy_text = _read_policy("violating_policy.txt")

    return RegComplianceObservation(
        regulation_text=regulation_text,
        policy_text=policy_text,
        task_id="medium",
        article_refs=["Article 5", "Article 6", "Article 13"],
        instructions=(
            "Perform a full GDPR audit of the privacy policy below. "
            "Identify ALL violations against Articles 5, 6, and 13. "
            "Use violation IDs like: ART5-PURPOSE, ART5-MINIMISATION, ART5-RETENTION, "
            "ART6-CONSENT, ART6-LAWFUL-BASIS, ART13-TRANSPARENCY, ART13-RETENTION-NOT-STATED. "
            "List every violation you find."
        ),
        context={"source": "sample_policy", "difficulty": "medium"},
    )


def _ground_truth_medium() -> dict[str, Any]:
    return {
        "violation_ids": [
            "ART5-PURPOSE",
            "ART5-MINIMISATION",
            "ART6-LAWFUL-BASIS",
            "ART13-TRANSPARENCY",
            "ART5-RETENTION",
        ],
        "severity": "high",
    }


def grade_medium(action: RegComplianceAction, ground_truth: dict[str, Any]) -> float:
    """Grade medium task: concept-based F1 over 5 GDPR concepts.

    EXPECTED_CONCEPTS: 5 groups of keywords. A concept is matched if ANY
    predicted violation ID contains ANY keyword from that concept group.

    precision = valid_flags / len(predicted)   (or 0.05 if none predicted)
    recall    = matched_concepts / 5
    f1        = 2*p*r/(p+r)                    (or 0.05 if both zero)
    score     = safe_score(f1 * 0.90 + 0.05)   → maps to [0.05, 0.95]
    """
    EXPECTED_CONCEPTS = [
        ["PURPOSE", "LIMITATION", "ART5-PURPOSE"],      # Art 5 purpose limitation
        ["MINIMIS", "ART5-MINIM"],                       # Art 5 data minimisation
        ["LAWFUL", "BASIS", "ART6", "CONSENT"],          # Art 6 lawful basis / consent
        ["TRANSPARENT", "ART13", "ART12"],               # Art 13 transparency
        ["RETENTION", "STORAGE", "ART5-RETAIN"],         # Art 5 storage limitation
    ]

    predicted = [v.upper() for v in action.violation_ids]

    # Count matched concepts (recall)
    matched_concepts = 0
    for concept_keywords in EXPECTED_CONCEPTS:
        for pred in predicted:
            if any(kw in pred for kw in concept_keywords):
                matched_concepts += 1
                break

    total_concepts = len(EXPECTED_CONCEPTS)  # 5

    recall = matched_concepts / total_concepts if total_concepts else 0.05

    # Precision: fraction of predicted flags that map to a known concept
    valid_flags = 0
    for pred in predicted:
        for concept_keywords in EXPECTED_CONCEPTS:
            if any(kw in pred for kw in concept_keywords):
                valid_flags += 1
                break

    precision = valid_flags / len(predicted) if predicted else 0.05
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.05

    raw = f1 * 0.90 + 0.05  # maps f1=0 → 0.05, f1=1 → 0.95
    return safe_score(raw)


# ---------------------------------------------------------------------------
# TASK 3 — hard
# ---------------------------------------------------------------------------

def _build_hard_observation(gdpr: dict[str, Any]) -> RegComplianceObservation:
    parts: list[str] = []
    for art_num in ("5", "6", "7", "12", "13", "17"):
        entry = gdpr.get(art_num, {})
        text = entry.get("full_text", "") if isinstance(entry, dict) else str(entry)
        if text:
            parts.append(f"=== Article {art_num} ===\n{text}")
    regulation_text = "\n\n".join(parts)

    policy_v1 = _read_policy("violating_policy.txt")
    policy_v2 = _read_policy("borderline_policy.txt")

    policy_text = (
        "POLICY VERSION 1 (original — has known violations):\n"
        f"{policy_v1}\n\n"
        "POLICY VERSION 2 (updated — some violations may be fixed):\n"
        f"{policy_v2}"
    )

    return RegComplianceObservation(
        regulation_text=regulation_text,
        policy_text=policy_text,
        task_id="hard",
        article_refs=["Article 5", "Article 6", "Article 7", "Article 12", "Article 13", "Article 17"],
        instructions=(
            "Compare the two policy versions above. "
            "1. List ALL GDPR violation IDs present in EITHER version. "
            "2. In your explanation, state which violations were FIXED between v1 and v2. "
            "3. In fix_suggestion, provide a concrete remediation plan (minimum 100 chars) "
            "for violations that REMAIN in v2. "
            "Use IDs like: ART5-PURPOSE, ART5-RETENTION, ART6-CONSENT, ART6-LAWFUL-BASIS, "
            "ART13-TRANSPARENCY, ART17-ERASURE."
        ),
        context={
            "v1_source": "violating_policy.txt",
            "v2_source": "borderline_policy.txt",
            "difficulty": "hard",
        },
    )


def _ground_truth_hard() -> dict[str, Any]:
    return {
        "fixed_violations": ["ART6-LAWFUL-BASIS"],
        "new_violations": [],
        "remaining_violations": ["ART5-RETENTION", "ART13-TRANSPARENCY"],
        "severity": "high",
    }


def grade_hard(action: RegComplianceAction, ground_truth: dict[str, Any]) -> float:
    """Grade hard task: 3-dimensional weighted score.

    dim1 (0.40): violations found — 0.05 if none, else min(0.90, count/3 * 0.90)
    dim2 (0.35): fix_suggestion quality — 0.90 if len > 80 else 0.25
    dim3 (0.25): explanation quality   — 0.90 if len > 60 else 0.20

    raw  = 0.40*dim1 + 0.35*dim2 + 0.25*dim3
    score = safe_score(raw)
    """
    violations = action.violation_ids
    fix_suggestion = action.fix_suggestion or ""
    explanation = action.explanation or ""

    # dim1: violation detection
    if not violations:
        dim1 = 0.05
    else:
        dim1 = min(0.90, len(violations) / 3 * 0.90)

    # dim2: fix suggestion quality
    dim2 = 0.90 if len(fix_suggestion) > 80 else 0.25

    # dim3: explanation quality
    dim3 = 0.90 if len(explanation) > 60 else 0.20

    raw = 0.40 * dim1 + 0.35 * dim2 + 0.25 * dim3
    return safe_score(raw)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_CONFIGS: dict[str, dict[str, Any]] = {
    "easy": {
        "id": "easy",
        "name": "Single GDPR clause check",
        "description": "Check one policy clause against GDPR Article 6 lawful basis.",
        "difficulty": "easy",
        "max_steps": 1,
        "success_threshold": 0.5,
        "article_refs": ["Article 6"],
        "build_observation": _build_easy_observation,
        "ground_truth": _ground_truth_easy,
        "grader": grade_easy,
    },
    "medium": {
        "id": "medium",
        "name": "Full privacy policy GDPR audit",
        "description": "Audit an entire violating privacy policy against Articles 5, 6, 13.",
        "difficulty": "medium",
        "max_steps": 1,
        "success_threshold": 0.5,
        "article_refs": ["Article 5", "Article 6", "Article 13"],
        "build_observation": _build_medium_observation,
        "ground_truth": _ground_truth_medium,
        "grader": grade_medium,
    },
    "hard": {
        "id": "hard",
        "name": "Policy version delta and remediation",
        "description": "Compare two policy versions, identify remaining violations, suggest fixes.",
        "difficulty": "hard",
        "max_steps": 1,
        "success_threshold": 0.5,
        "article_refs": ["Article 5", "Article 6", "Article 7", "Article 12", "Article 13", "Article 17"],
        "build_observation": _build_hard_observation,
        "ground_truth": _ground_truth_hard,
        "grader": grade_hard,
    },
}


def get_task_config(task_id: str) -> dict[str, Any]:
    """Retrieve a task config by ID. Raises ValueError for unknown tasks."""
    if task_id not in TASK_CONFIGS:
        raise ValueError(
            f"Unknown task '{task_id}'. Valid options: {list(TASK_CONFIGS.keys())}"
        )
    return TASK_CONFIGS[task_id]
