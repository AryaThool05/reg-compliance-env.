"""
client.py — EnvClient for interacting with a running RegComplianceEnv server.

Wraps the HTTP API exposed by server/app.py into a simple Python interface.
"""

from __future__ import annotations

from typing import Any

import httpx

try:
    from models import RegComplianceAction, RegComplianceObservation
except ImportError:
    from .models import RegComplianceAction, RegComplianceObservation


class EnvClient:
    """HTTP client for the RegComplianceEnv server API.

    Usage::

        client = EnvClient("http://localhost:7860")
        obs = client.reset("easy")
        result = client.step(action)
        print(result["reward"])
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ---- API methods -------------------------------------------------------

    def health(self) -> dict[str, str]:
        """Check server health. Returns {"status": "healthy"} when OK."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def reset(self, task_id: str = "easy") -> dict[str, Any]:
        """Reset the environment for a given task.

        Args:
            task_id: One of "easy", "medium", "hard".

        Returns:
            Dict with "observation" and "info" keys.
        """
        resp = self._client.post("/reset", json={"task_id": task_id})
        resp.raise_for_status()
        return resp.json()

    def step(self, action: RegComplianceAction | dict[str, Any]) -> dict[str, Any]:
        """Submit an action and receive reward/done/info.

        Args:
            action: A RegComplianceAction instance or equivalent dict.

        Returns:
            Dict with "observation", "reward", "done", "info" keys.
        """
        if isinstance(action, RegComplianceAction):
            payload = action.model_dump()
        else:
            payload = dict(action)

        resp = self._client.post("/step", json=payload)
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict[str, Any]:
        """Get current environment state."""
        resp = self._client.get("/state")
        resp.raise_for_status()
        return resp.json()

    # ---- Context manager support ------------------------------------------

    def __enter__(self) -> "EnvClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
