---
title: Reg Compliance Env
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# RegComplianceEnv

A GDPR regulatory compliance checker environment for the Meta PyTorch x Scaler OpenEnv hackathon. An AI agent audits privacy policies against EU GDPR articles and identifies violations across three difficulty tiers.

## Overview

RegComplianceEnv presents an agent with real GDPR regulation text and company privacy policies, then scores the agent's ability to identify violations, assess severity, and suggest remediation. All scoring uses `safe_score()` to ensure rewards are strictly in **(0.05, 0.95)** — never exactly 0 or 1.

## Task Tiers

| Tier | Task | Articles | Grader |
|------|------|----------|--------|
| **Easy** | Single clause check | Article 6 | Keyword-based: ART6/CONSENT/LAWFUL keywords |
| **Medium** | Full policy audit | Articles 5, 6, 13 | Concept-based F1 over 5 GDPR concepts |
| **Hard** | Policy delta analysis | Articles 5–17 | 3-dim weighted: violations + fix quality + explanation |

## Scoring

All rewards are strictly in **(0.05, 0.95)** using `safe_score()`:

```python
def safe_score(raw: float) -> float:
    if raw <= 0.0: return 0.05
    if raw >= 1.0: return 0.95
    return round(min(0.95, max(0.05, raw)), 4)
```

- **Easy**: `safe_score(0.45 * violation_found + 0.45 * article_correct + 0.05)`
- **Medium**: `safe_score(f1 * 0.90 + 0.05)` over 5 GDPR concept groups
- **Hard**: `safe_score(0.40*dim1 + 0.35*dim2 + 0.25*dim3)` (violations + fix suggestion + explanation)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check → `{"status": "healthy"}` |
| `GET` | `/` | Service info |
| `POST` | `/reset` | Reset with `{"task_id": "easy|medium|hard"}` |
| `POST` | `/step` | Submit action with violation_ids, severity, explanation |
| `GET` | `/state` | Current environment state |

## Running Locally

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Running Inference

```bash
export HF_TOKEN=hf_your_token_here
export MODEL_NAME=gpt-4.1-mini          # optional, this is the default
export API_BASE_URL=https://api.openai.com/v1  # optional, this is the default
python inference.py
```

## Docker

```bash
docker build -t reg-compliance-env .
docker run -p 7860:7860 -e HF_TOKEN=hf_... reg-compliance-env
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## File Structure

```
reg-compliance-env/
├── inference.py          ← Main inference script
├── models.py             ← RegComplianceObservation, RegComplianceAction, RegComplianceState
├── task_definitions.py   ← All 3 tasks + graders + safe_score
├── client.py             ← EnvClient HTTP wrapper
├── openenv.yaml          ← Environment manifest
├── pyproject.toml
├── requirements.txt
├── __init__.py           ← Package exports
├── data/
│   ├── gdpr_articles.json        ← Static GDPR text (Art 5,6,7,12,13,17)
│   └── sample_policies/
│       ├── clean_policy.txt
│       ├── violating_policy.txt
│       └── borderline_policy.txt
├── server/
│   ├── __init__.py
│   ├── app.py            ← FastAPI using create_app pattern
│   └── environment.py    ← RegComplianceEnvironment
└── tests/
    ├── test_environment.py
    ├── test_grader_guards.py
    ├── test_inference_logging.py
    └── test_inference_policy.py
```

## Author

Arya Thool
