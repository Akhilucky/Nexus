"""Core Routing Logic — orchestrates intent → scoring → selection.

Routes an incoming request through:
  1. Intent classification
  2. Decision engine scoring
  3. Top-tool selection
  4. Telemetry recording
"""

from __future__ import annotations

import logging
import time

from nexus.app.decision_engine import DecisionEngine
from nexus.app.telemetry import TelemetryStore
from nexus.models.schemas import (
    ExecutionRecord,
    IntentCategory,
    RouteRequest,
    RouteResponse,
    ScoredTool,
)

logger = logging.getLogger(__name__)


class Router:
    """Top-level routing controller."""

    def __init__(
        self,
        decision_engine: DecisionEngine,
        telemetry: TelemetryStore,
    ) -> None:
        self.engine = decision_engine
        self.telemetry = telemetry

    def route(self, request: RouteRequest) -> RouteResponse:
        """Score tools and return the best match for the incoming query."""
        start = time.perf_counter()

        # 1. Classify intent (informational — enriches logs)
        intent, intent_scores = self.engine.intent.classify(request.query)
        logger.info("Intent: %s  scores=%s", intent.value, intent_scores)

        # 2. Score candidate tools
        scored: list[ScoredTool] = self.engine.score_tools(
            query=request.query,
            tags_hint=request.tags_hint,
            max_results=request.max_results,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        if not scored:
            resp = RouteResponse(
                selected_tool="none",
                confidence=0.0,
                reasoning_trace=[],
            )
            self._record(resp, request.query, success=True, latency_ms=elapsed_ms)
            return resp

        best = scored[0]
        resp = RouteResponse(
            selected_tool=best.tool_name,
            confidence=best.score,
            reasoning_trace=scored,
        )

        self._record(resp, request.query, success=True, latency_ms=elapsed_ms)
        return resp

    # ------------------------------------------------------------------
    # Feedback ingestion
    # ------------------------------------------------------------------

    def record_execution_result(
        self,
        request_id: str,
        success: bool,
        latency_ms: float,
        user_satisfaction: float | None = None,
        error_message: str | None = None,
    ) -> None:
        """Called after the external caller actually executes the tool.

        Updates telemetry + tool reputation through the memory/learning loop.
        """
        record = self.telemetry.get_by_request_id(request_id)
        if record is None:
            logger.warning("No telemetry record found for request_id=%s", request_id)
            return

        updated = record.model_copy(
            update={
                "success": success,
                "latency_ms": latency_ms,
                "user_satisfaction": user_satisfaction,
                "error_message": error_message,
            }
        )
        self.telemetry.update(updated)

        # Update tool stats in registry
        tool = self.engine.registry.get(record.tool_name)
        if tool:
            new_calls = tool.total_calls + 1
            new_successes = tool.total_successes + (1 if success else 0)
            self.engine.registry.update_tool(
                tool.name,
                total_calls=new_calls,
                total_successes=new_successes,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _record(
        self,
        resp: RouteResponse,
        query: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """Log the routing decision to telemetry."""
        if resp.selected_tool == "none":
            return

        best = resp.reasoning_trace[0] if resp.reasoning_trace else None
        if best is None:
            return

        record = ExecutionRecord(
            request_id=resp.request_id,
            tool_name=best.tool_name,
            tool_id=best.tool_id,
            query=query,
            success=success,
            latency_ms=latency_ms,
        )
        self.telemetry.add(record)
