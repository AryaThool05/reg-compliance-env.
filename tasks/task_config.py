"""
Task configuration registry for RegComplianceEnv.

Maps task difficulty levels to their metadata used by the OpenEnv harness.
"""

TASK_CONFIGS: dict[str, dict[str, object]] = {
    "easy": {
        "name": "single-clause-check",
        "benchmark": "reg-compliance-env",
        "max_steps": 1,
    },
    "medium": {
        "name": "full-policy-audit",
        "benchmark": "reg-compliance-env",
        "max_steps": 1,
    },
    "hard": {
        "name": "policy-delta-remediation",
        "benchmark": "reg-compliance-env",
        "max_steps": 1,
    },
}
