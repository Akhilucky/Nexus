"""Pydantic schemas for NEXUS — the LLM Decision Router."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class IntentCategory(str, Enum):
    retrieval = "retrieval"
    computation = "computation"
    visualization = "visualization"
    automation = "automation"
    summarization = "summarization"
    unknown = "unknown"


class SecurityLevel(str, Enum):
    public = "public"
    internal = "internal"
    confidential = "confidential"
    restricted = "restricted"


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class ToolRegistration(BaseModel):
    """Payload sent when a new tool is registered."""

    name: str = Field(..., min_length=1, max_length=128, description="Unique tool name")
    description: str = Field(..., min_length=1, description="Human-readable description")
    input_schema: dict[str, Any] = Field(default_factory=dict, description="Expected input schema")
    latency_ms: float = Field(default=100.0, ge=0, description="Typical latency (ms)")
    cost: float = Field(default=0.0, ge=0, description="Cost per invocation (USD)")
    reliability: float = Field(default=0.95, ge=0.0, le=1.0, description="Historical reliability 0–1")
    security_level: SecurityLevel = Field(default=SecurityLevel.internal)
    tags: list[str] = Field(default_factory=list, description="Categorical tags")


class Tool(ToolRegistration):
    """Internal representation stored in the registry."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Mutable runtime stats
    reputation: float = Field(default=0.5, ge=0.0, le=1.0, description="Learned reputation score")
    total_calls: int = Field(default=0, ge=0)
    total_successes: int = Field(default=0, ge=0)
    embedding: list[float] | None = Field(default=None, exclude=True, description="Cached embedding vector")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class RouteRequest(BaseModel):
    """Incoming request to be routed."""

    query: str = Field(..., min_length=1, description="User query / task description")
    tags_hint: list[str] | None = Field(default=None, description="Optional tag hints to narrow search")
    max_results: int = Field(default=3, ge=1, le=10, description="Number of candidate tools to return")
    security_clearance: SecurityLevel = Field(
        default=SecurityLevel.internal,
        description="Maximum security level allowed for selected tools",
    )
    blocked_tags: list[str] | None = Field(
        default=None,
        description="Optional deny-list of tags that tools must not include",
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Optional allow-list restricting routing to specific tool names",
    )
    min_reliability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum reliability required for candidate tools",
    )


class ScoredTool(BaseModel):
    """A single scored candidate in the routing response."""

    tool_name: str
    tool_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: dict[str, float] = Field(default_factory=dict, description="Per-factor scores")


class RouteResponse(BaseModel):
    """Response returned by the /route endpoint."""

    selected_tool: str
    confidence: float
    reasoning_trace: list[ScoredTool] = Field(default_factory=list)
    intent: IntentCategory = Field(default=IntentCategory.unknown)
    intent_scores: dict[str, float] = Field(default_factory=dict)
    policy_trace: dict[str, Any] = Field(default_factory=dict)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class ExecutionRecord(BaseModel):
    """Logged after every routed tool execution."""

    request_id: str
    tool_name: str
    tool_id: str
    query: str
    success: bool
    latency_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    user_satisfaction: float | None = Field(default=None, ge=0.0, le=1.0)
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class ToolMetrics(BaseModel):
    """Aggregate metrics for a single tool."""

    tool_name: str
    total_calls: int
    total_successes: int
    avg_latency_ms: float
    reputation: float
    success_rate: float


class SystemMetrics(BaseModel):
    """Overall system health."""

    total_routes: int
    tool_count: int
    tools: list[ToolMetrics]
