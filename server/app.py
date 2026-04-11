"""
server/app.py — FastAPI server for RegComplianceEnv.

Uses the openenv.core create_app pattern where available.
Falls back to a standalone FastAPI app for local development.

Exposes:
  POST /reset  → HTTP 200 with observation JSON
  POST /step   → HTTP 200 with {observation, reward, done, info}
  GET  /state  → HTTP 200 with state dict
  GET  /health → HTTP 200 with {"status": "healthy"}
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from ..models import RegComplianceAction, RegComplianceObservation
except ImportError:
    from models import RegComplianceAction, RegComplianceObservation

from .environment import RegComplianceEnvironment

# ---------------------------------------------------------------------------
# Try openenv.core create_app pattern first
# ---------------------------------------------------------------------------

try:
    from openenv.core.env_server import create_app as _create_app

    app = _create_app(
        RegComplianceEnvironment,
        RegComplianceAction,
        RegComplianceObservation,
        env_name="reg-compliance-env",
    )

except Exception:
    # ---------------------------------------------------------------------------
    # Standalone FastAPI fallback (always works, even without openenv-core)
    # ---------------------------------------------------------------------------

    from contextlib import asynccontextmanager

    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    class ResetRequest(BaseModel):
        task_id: str = Field(default="easy", description="Task difficulty: easy, medium, hard")

    class StepRequest(BaseModel):
        violation_ids: list[str] = Field(default_factory=list)
        severity: str = Field(default="none")
        explanation: str = Field(default="")
        fix_suggestion: str = Field(default="")

    # Global environment instance
    _env = RegComplianceEnvironment()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # Pre-load GDPR cache
        RegComplianceEnvironment._get_gdpr_cache()
        yield

    app = FastAPI(
        title="RegComplianceEnv",
        description="GDPR compliance checker OpenEnv environment API",
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

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"status": "ok", "env": "reg-compliance-env", "version": "1.0.0"}

    @app.post("/reset")
    async def reset(body: Optional[ResetRequest] = None) -> dict[str, Any]:
        task_id = body.task_id if body else "easy"
        try:
            obs = _env.reset(task_id=task_id)
            return {"observation": obs, "info": {"task_id": task_id}}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/step")
    async def step(body: StepRequest) -> dict[str, Any]:
        try:
            result = _env.step(body.model_dump())
            return result
        except TypeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/state")
    async def state() -> dict[str, Any]:
        return _env.state()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
