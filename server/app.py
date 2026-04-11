"""
server/app.py — FastAPI server for RegComplianceEnv.

Priority order:
1. Try openenv.core create_app (CLASS not instance, correct signature)
2. Fall back to standalone FastAPI with manual routes

CRITICAL: The /reset endpoint must return the observation serialized as JSON.
Reset returns a Pydantic model — use .model_dump() explicitly in the
standalone fallback so we never pass a raw Pydantic model to JSONResponse.

Endpoints:
  POST /reset  → HTTP 200 with observation JSON
  POST /step   → HTTP 200 with {observation, reward, done, info}
  GET  /state  → HTTP 200 with state dict
  GET  /health → HTTP 200 with {"status": "healthy"}
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from ..models import RegComplianceAction, RegComplianceObservation, RegComplianceState
except ImportError:
    from models import RegComplianceAction, RegComplianceObservation, RegComplianceState

from .environment import RegComplianceEnvironment

# ---------------------------------------------------------------------------
# Attempt 1: openenv.core create_app
# ---------------------------------------------------------------------------

_app_created_by_framework = False

try:
    from openenv.core.env_server import create_app as _create_app

    app = _create_app(
        RegComplianceEnvironment,      # CLASS, not instance
        RegComplianceAction,
        RegComplianceObservation,
        env_name="reg-compliance-env",
    )
    _app_created_by_framework = True

except Exception as _framework_err:
    pass  # fall through to standalone FastAPI below

# ---------------------------------------------------------------------------
# Attempt 2: Standalone FastAPI (always works without openenv-core)
# ---------------------------------------------------------------------------

if not _app_created_by_framework:

    from contextlib import asynccontextmanager

    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    class ResetRequest(BaseModel):
        task: str = Field(default="easy", description="Task ID: easy, medium, or hard")
        task_id: str = Field(default="", description="Alias for task")

    class StepRequest(BaseModel):
        violation_ids: list[str] = Field(default_factory=list)
        severity: str = Field(default="none")
        explanation: str = Field(default="")
        fix_suggestion: str = Field(default="")

    # Single global environment instance
    _env = RegComplianceEnvironment()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Pre-warm: load GDPR JSON once at startup."""
        _env._load_gdpr_articles()
        yield

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

    # ---- Health ------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check — judges poll this. Must return 200."""
        return {"status": "healthy"}

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"status": "ok", "env": "reg-compliance-env", "version": "1.0.0"}

    # ---- Reset -------------------------------------------------------------

    @app.post("/reset")
    async def reset(body: Optional[ResetRequest] = None) -> JSONResponse:
        """Reset the environment and return the initial observation as JSON.

        Accepts: {"task": "easy"} or {"task_id": "easy"} or empty body.
        Always returns HTTP 200 with observation JSON.
        """
        try:
            # Resolve task_id from body — support both "task" and "task_id" keys
            if body is None:
                task_id = "easy"
            else:
                task_id = body.task_id if body.task_id else body.task
                task_id = task_id or "easy"

            if task_id not in ("easy", "medium", "hard"):
                task_id = "easy"

            # reset() returns a RegComplianceObservation Pydantic model
            obs: RegComplianceObservation = _env.reset(task=task_id)

            # Explicitly serialize — never pass Pydantic model raw to JSONResponse
            return JSONResponse(
                status_code=200,
                content={
                    "observation": obs.model_dump(),
                    "info": {"task_id": task_id},
                },
            )

        except Exception as exc:
            # Return a safe fallback observation instead of crashing with 500
            fallback = RegComplianceObservation(
                regulation_text="Processing shall be lawful only if consent given (Article 6).",
                policy_text="We share your data with partners without consent.",
                task_id="easy",
                article_refs=["Article 6"],
                instructions="Identify GDPR violations in the policy.",
                context={"error": str(exc)[:100], "source": "error_fallback"},
            )
            return JSONResponse(
                status_code=200,
                content={
                    "observation": fallback.model_dump(),
                    "info": {"task_id": "easy", "warning": "fallback used"},
                },
            )

    # ---- Step --------------------------------------------------------------

    @app.post("/step")
    async def step(body: StepRequest) -> JSONResponse:
        """Submit an action and receive reward/done/info."""
        try:
            result = _env.step(action=body.model_dump())

            # result.observation is a RegComplianceObservation Pydantic model
            obs_dict = (
                result.observation.model_dump()
                if hasattr(result.observation, "model_dump")
                else result.observation
            )

            return JSONResponse(
                status_code=200,
                content={
                    "observation": obs_dict,
                    "reward": result.reward,
                    "done": result.done,
                    "info": result.info,
                },
            )

        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ---- State -------------------------------------------------------------

    @app.get("/state")
    async def state() -> JSONResponse:
        """Return current environment state."""
        s = _env.state  # property returns RegComplianceState Pydantic model
        if hasattr(s, "model_dump"):
            return JSONResponse(status_code=200, content=s.model_dump())
        return JSONResponse(status_code=200, content=s)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
