"""
server/app.py — Pure FastAPI server for RegComplianceEnv.

NO openenv.core imports. NO create_app. Full manual control.

Endpoints:
  GET  /         → service info
  GET  /health   → {"status": "healthy"}
  POST /reset    → observation JSON (always HTTP 200)
  POST /step     → {observation, reward, done, info} (always HTTP 200)
  GET  /state    → state dict (always HTTP 200)
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from server.environment import RegComplianceEnvironment
except ImportError:
    from environment import RegComplianceEnvironment

try:
    from models import RegComplianceAction
except ImportError:
    from models import RegComplianceAction  # same, but satisfies linter

# ---------------------------------------------------------------------------
# Global environment instance — created once at module load
# ---------------------------------------------------------------------------

_env = RegComplianceEnvironment()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: env already initialised above. Shutdown: nothing to clean up."""
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RegComplianceEnv",
    description="GDPR compliance checker — OpenEnv environment for hackathon",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: Optional[str] = "easy"
    task_id: Optional[str] = None   # alias — some callers send task_id


class StepRequest(BaseModel):
    violation_ids: list = []
    severity: str = "none"
    explanation: str = ""
    fix_suggestion: str = ""


# ---------------------------------------------------------------------------
# Safe Score Helper
# ---------------------------------------------------------------------------

def _safe(r: float) -> float:
    """Ensure reward is strictly between 0.0 and 1.0."""
    try:
        r = float(r)
    except Exception:
        return 0.05
    if r <= 0.0:
        return 0.05
    if r >= 1.0:
        return 0.95
    return r

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok", "env": "reg-compliance-env", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check — must return 200. Judges poll this."""
    return {"status": "healthy", "service": "reg-compliance-env"}


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    """Reset the environment and return the initial observation."""
    try:
        # Resolve task from body
        task = "easy"
        if request is not None:
            task = request.task_id or request.task or "easy"
        if task not in ("easy", "medium", "hard"):
            task = "easy"

        obs = _env.reset(task=task)

        if hasattr(obs, "model_dump"):
            return JSONResponse(content=obs.model_dump(), status_code=200)
        return JSONResponse(content=obs if isinstance(obs, dict) else {}, status_code=200)

    except Exception as exc:
        return JSONResponse(content={
            "observation": {
                "regulation_text": "GDPR compliance environment",
                "policy_text": "",
                "task_id": "easy",
                "article_refs": ["Article 6"],
                "instructions": "Identify GDPR violations",
                "context": {},
                "reward": 0.5,
                "done": False
            },
            "reward": 0.05,       
            "done": True,
            "info": {"error": str(exc)[:200]}
        }, status_code=200)


@app.post("/step")
async def step(request: StepRequest):
    """Submit an action and receive reward/done/info."""
    try:
        action = RegComplianceAction(
            violation_ids=request.violation_ids,
            severity=request.severity,
            explanation=request.explanation,
            fix_suggestion=request.fix_suggestion,
        )
        result = _env.step(action)

        obs = getattr(result, "observation", None)
        obs_dict = (
            obs.model_dump() if hasattr(obs, "model_dump")
            else (obs if isinstance(obs, dict) else {})
        )

        reward = _safe(float(getattr(result, 'reward', 0.05)))

        return JSONResponse(
            content={
                "observation": obs_dict,
                "reward": reward,
                "done": bool(getattr(result, "done", True)),
                "info": getattr(result, "info", {}),
            },
            status_code=200,
        )

    except Exception as exc:
        return JSONResponse(content={
            "observation": {
                "regulation_text": "GDPR compliance environment",
                "policy_text": "",
                "task_id": "easy",
                "article_refs": ["Article 6"],
                "instructions": "Identify GDPR violations",
                "context": {},
                "reward": 0.5,
                "done": False
            },
            "reward": 0.05,       
            "done": True,
            "info": {"error": str(exc)[:200]}
        }, status_code=200)


@app.get("/state")
async def state():
    """Return current environment state."""
    try:
        s = _env.state 
        if hasattr(s, "model_dump"):
            return JSONResponse(content=s.model_dump(), status_code=200)
        return JSONResponse(
            content=s if isinstance(s, dict) else {},
            status_code=200,
        )
    except Exception as exc:
        return JSONResponse(content={
            "task_id": "easy",
            "step_count": 0,
            "done": False,
            "episode_id": "",
            "error": str(exc)[:200],
            "reward": 0.05,
        }, status_code=200)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
