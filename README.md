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

A GDPR regulatory compliance checker environment where an AI agent audits privacy policies against EU GDPR articles and identifies violations.

## Overview

RegComplianceEnv is an OpenEnv-compatible environment built for the GDPR compliance checker hackathon. It presents an agent with real GDPR regulation text and company privacy policies, then scores the agent's ability to identify violations, assess severity, and suggest fixes.

## Task Tiers

| Tier | Task | Description |
|------|------|-------------|
| **Easy** | Single clause check | Identify a consent violation against Article 6 |
| **Medium** | Full policy audit | Audit an entire privacy policy against Articles 5, 6, 13 |
| **Hard** | Policy delta analysis | Compare two policy versions, find fixed/remaining violations, suggest remediation |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check |
| `POST` | `/reset` | Reset environment with `{"task": "easy\|medium\|hard"}` |
| `POST` | `/step` | Submit action with violation_ids, severity, explanation |
| `GET` | `/state` | Current environment state |

## Scoring

Rewards are computed using F1-based scoring with partial credit:
- **Easy**: +0.5 for finding any violation, +0.5 for citing the correct article
- **Medium**: Set-based precision/recall/F1 over violation IDs
- **Hard**: Weighted scoring across fixed violations (0.35), new violations (0.35), and fix suggestion quality (0.30)

All scores are clamped to [0.0, 1.0].

## Running Locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Running Inference

```bash
python inference.py
```

## Docker

```bash
docker build -t reg-compliance-env .
docker run -p 7860:7860 reg-compliance-env
```

## Author

Arya Thool
