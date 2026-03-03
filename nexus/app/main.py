"""FastAPI entrypoint for NEXUS — LLM Decision Router.

Endpoints:
    POST  /tools/register   — Register or update a tool
    GET   /tools             — List all registered tools
    DELETE /tools/{name}     — Remove a tool
    POST  /route             — Route a user query to the best tool
    POST  /feedback          — Report execution outcome (learning loop)
    GET   /metrics           — System-wide telemetry metrics
    POST  /admin/recalculate — Recompute all reputations from history
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from nexus.app.decision_engine import DecisionEngine
from nexus.app.intent import IntentClassifier
from nexus.app.memory import MemoryManager
from nexus.app.registry import ToolRegistry
from nexus.app.router import Router
from nexus.app.telemetry import TelemetryStore
from nexus.models.schemas import (
    RouteRequest,
    RouteResponse,
    SystemMetrics,
    Tool,
    ToolRegistration,
)

logger = logging.getLogger("nexus")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Singleton components (created at startup)
# ---------------------------------------------------------------------------

_registry: ToolRegistry | None = None
_intent: IntentClassifier | None = None
_engine: DecisionEngine | None = None
_telemetry: TelemetryStore | None = None
_router: Router | None = None
_memory: MemoryManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _registry, _intent, _engine, _telemetry, _router, _memory

    logger.info("NEXUS starting up …")
    _registry = ToolRegistry()
    _intent = IntentClassifier()
    _engine = DecisionEngine(_registry, _intent)
    _telemetry = TelemetryStore()
    _router = Router(_engine, _telemetry)
    _memory = MemoryManager(_registry, _telemetry)
    logger.info("NEXUS ready — %d tools loaded", len(_registry.list_all()))

    yield  # app runs here

    logger.info("NEXUS shutting down.")


app = FastAPI(
    title="NEXUS — LLM Decision Router",
    description="A control plane for tool-using AI systems",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Tool Registry endpoints
# ---------------------------------------------------------------------------

@app.post("/tools/register", response_model=Tool)
async def register_tool(payload: ToolRegistration):
    assert _registry is not None
    tool = _registry.register(payload)
    logger.info("Registered tool: %s", tool.name)
    return tool


@app.get("/tools", response_model=list[Tool])
async def list_tools():
    assert _registry is not None
    return _registry.list_all()


@app.get("/tools/{name}", response_model=Tool)
async def get_tool(name: str):
    assert _registry is not None
    tool = _registry.get(name)
    if tool is None:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
    return tool


@app.delete("/tools/{name}")
async def delete_tool(name: str):
    assert _registry is not None
    removed = _registry.remove(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
    return {"removed": name}


# ---------------------------------------------------------------------------
# Routing endpoint
# ---------------------------------------------------------------------------

@app.post("/route", response_model=RouteResponse)
async def route_request(payload: RouteRequest):
    assert _router is not None
    return _router.route(payload)


# ---------------------------------------------------------------------------
# Feedback / Learning endpoint
# ---------------------------------------------------------------------------

class FeedbackPayload(BaseModel):
    request_id: str
    success: bool
    latency_ms: float = Field(ge=0)
    user_satisfaction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    error_message: Optional[str] = None


@app.post("/feedback")
async def submit_feedback(payload: FeedbackPayload):
    assert _router is not None and _memory is not None

    _router.record_execution_result(
        request_id=payload.request_id,
        success=payload.success,
        latency_ms=payload.latency_ms,
        user_satisfaction=payload.user_satisfaction,
        error_message=payload.error_message,
    )

    # Also update reputation via the memory/learning loop
    record = _telemetry.get_by_request_id(payload.request_id)  # type: ignore[union-attr]
    if record:
        _memory.update_reputation(record.tool_name, payload.success)

    return {"status": "recorded", "request_id": payload.request_id}


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------

@app.get("/metrics", response_model=SystemMetrics)
async def get_metrics():
    assert _registry is not None and _telemetry is not None
    tools = _registry.list_all()
    metrics = _telemetry.system_metrics([t.name for t in tools])
    # Enrich with registry reputation
    for tm in metrics.tools:
        t = _registry.get(tm.tool_name)
        if t:
            tm.reputation = t.reputation
    return metrics


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

@app.post("/admin/recalculate")
async def recalculate_reputations():
    assert _memory is not None
    results = _memory.recalculate_all()
    return {"reputations": results}
