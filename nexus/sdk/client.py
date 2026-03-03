"""NEXUS SDK Client — how tools integrate with the NEXUS router.

Provides a simple Python client for:
  - Registering tools
  - Routing queries
  - Submitting feedback
  - Fetching metrics
"""

from __future__ import annotations

from typing import Any, Optional

import httpx


class NexusClient:
    """HTTP client for the NEXUS Decision Router API."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=30.0)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "NexusClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict:
        r = self._client.get("/health")
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Tool Registration
    # ------------------------------------------------------------------

    def register_tool(
        self,
        name: str,
        description: str,
        *,
        input_schema: dict[str, Any] | None = None,
        latency_ms: float = 100.0,
        cost: float = 0.0,
        reliability: float = 0.95,
        security_level: str = "internal",
        tags: list[str] | None = None,
    ) -> dict:
        payload = {
            "name": name,
            "description": description,
            "input_schema": input_schema or {},
            "latency_ms": latency_ms,
            "cost": cost,
            "reliability": reliability,
            "security_level": security_level,
            "tags": tags or [],
        }
        r = self._client.post("/tools/register", json=payload)
        r.raise_for_status()
        return r.json()

    def list_tools(self) -> list[dict]:
        r = self._client.get("/tools")
        r.raise_for_status()
        return r.json()

    def get_tool(self, name: str) -> dict:
        r = self._client.get(f"/tools/{name}")
        r.raise_for_status()
        return r.json()

    def delete_tool(self, name: str) -> dict:
        r = self._client.delete(f"/tools/{name}")
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(
        self,
        query: str,
        *,
        tags_hint: list[str] | None = None,
        max_results: int = 3,
    ) -> dict:
        payload: dict[str, Any] = {"query": query, "max_results": max_results}
        if tags_hint:
            payload["tags_hint"] = tags_hint
        r = self._client.post("/route", json=payload)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def submit_feedback(
        self,
        request_id: str,
        success: bool,
        latency_ms: float,
        *,
        user_satisfaction: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "request_id": request_id,
            "success": success,
            "latency_ms": latency_ms,
        }
        if user_satisfaction is not None:
            payload["user_satisfaction"] = user_satisfaction
        if error_message is not None:
            payload["error_message"] = error_message
        r = self._client.post("/feedback", json=payload)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def metrics(self) -> dict:
        r = self._client.get("/metrics")
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    def recalculate_reputations(self) -> dict:
        r = self._client.post("/admin/recalculate")
        r.raise_for_status()
        return r.json()
