"""Tests for the Router."""

from pathlib import Path

import pytest

from nexus.app.decision_engine import DecisionEngine
from nexus.app.intent import IntentClassifier
from nexus.app.registry import ToolRegistry
from nexus.app.router import Router
from nexus.app.telemetry import TelemetryStore
from nexus.models.schemas import RouteRequest, ToolRegistration


@pytest.fixture
def components(tmp_path: Path):
    registry = ToolRegistry(store_path=tmp_path / "tools.json")
    registry.register(ToolRegistration(
        name="tool_a", description="Fast data fetcher", latency_ms=50, cost=0.001, reliability=0.9, tags=["data"]
    ))
    registry.register(ToolRegistration(
        name="tool_b", description="Heavy analytics engine", latency_ms=500, cost=0.05, reliability=0.95, tags=["analytics"]
    ))
    intent = IntentClassifier()
    engine = DecisionEngine(registry, intent)
    telemetry = TelemetryStore(log_path=tmp_path / "tel.jsonl")
    router = Router(engine, telemetry)
    return router, registry, telemetry


class TestRouter:
    def test_route_returns_response(self, components):
        router, _, _ = components
        resp = router.route(RouteRequest(query="get sales data"))
        assert resp.selected_tool != ""
        assert 0.0 <= resp.confidence <= 1.0
        assert resp.request_id

    def test_route_records_telemetry(self, components):
        router, _, telemetry = components
        resp = router.route(RouteRequest(query="analyze revenue"))
        records = telemetry.list_all()
        assert len(records) >= 1
        assert records[-1].request_id == resp.request_id

    def test_route_empty_registry(self, tmp_path: Path):
        registry = ToolRegistry(store_path=tmp_path / "empty.json")
        intent = IntentClassifier()
        engine = DecisionEngine(registry, intent)
        telemetry = TelemetryStore(log_path=tmp_path / "tel.jsonl")
        router = Router(engine, telemetry)

        resp = router.route(RouteRequest(query="anything"))
        assert resp.selected_tool == "none"
        assert resp.confidence == 0.0

    def test_feedback_updates_stats(self, components):
        router, registry, _ = components
        resp = router.route(RouteRequest(query="get data"))
        router.record_execution_result(
            request_id=resp.request_id,
            success=True,
            latency_ms=55.0,
        )
        tool = registry.get(resp.selected_tool)
        assert tool is not None
        assert tool.total_calls >= 1
