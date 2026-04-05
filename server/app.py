"""
FastAPI web server wrapping RegComplianceEnv for Hugging Face Spaces deployment.

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 7860

Or:
    python app.py
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from environment import RegComplianceEnv
from models import Action, Observation
from scraper import load_gdpr_cache


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = Field(default="easy", description="Task difficulty: easy, medium, or hard")


class StepRequest(BaseModel):
    violation_ids: list[str] = Field(default_factory=list)
    severity: str = Field(default="none")
    explanation: str = Field(default="")
    fix_suggestion: str | None = Field(default=None)


class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


# ---------------------------------------------------------------------------
# Global environment instance
# ---------------------------------------------------------------------------

env = RegComplianceEnv()


# ---------------------------------------------------------------------------
# Lifespan: pre-load GDPR cache on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load GDPR cache into memory at startup, clean up on shutdown."""
    # Startup
    _ = load_gdpr_cache()
    yield
    # Shutdown
    await env.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RegComplianceEnv",
    description="GDPR compliance checker OpenEnv environment API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins (required for HF Spaces)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint — basic service info."""
    return {
        "status": "ok",
        "env": "reg-compliance-env",
        "version": "1.0.0",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/reset")
async def reset(body: Optional[ResetRequest] = None) -> dict[str, Any]:
    """Reset the environment for a new episode.

    Accepts an optional ``task`` field (defaults to ``"easy"``).
    Body itself is optional — an empty POST defaults to task="easy".
    """
    task = body.task if body else "easy"
    try:
        result = await env.reset(task=task)
        return {
            "observation": result.observation.model_dump(),
            "info": result.info,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step")
async def step(body: StepRequest) -> StepResponse:
    """Execute one step with the given action.

    The request body is parsed into an ``Action`` model and passed to the
    environment.  Returns observation, reward, done flag, and info dict.
    """
    try:
        action = Action(
            violation_ids=body.violation_ids,
            severity=body.severity,
            explanation=body.explanation,
            fix_suggestion=body.fix_suggestion,
        )
        result = await env.step(action)
        return StepResponse(
            observation=result.observation.model_dump(),
            reward=result.reward,
            done=result.done,
            info=result.info,
        )
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state")
async def state() -> dict[str, Any]:
    """Return the current environment state."""
    return await env.state()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
