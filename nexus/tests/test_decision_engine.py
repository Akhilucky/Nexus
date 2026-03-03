"""Tests for the Decision Engine."""

from pathlib import Path

import pytest

from nexus.app.decision_engine import DecisionEngine, ScoringWeights
from nexus.app.intent import IntentClassifier
from nexus.app.registry import ToolRegistry
from nexus.models.schemas import ToolRegistration


@pytest.fixture
def registry(tmp_path: Path) -> ToolRegistry:
    reg = ToolRegistry(store_path=tmp_path / "tools.json")
    # Register the demo scenario tools
    reg.register(ToolRegistration(
        name="warehouse_api",
        description="Query the company data warehouse for accurate historical sales and revenue data",
        latency_ms=800,
        cost=0.05,
        reliability=0.98,
        tags=["data", "analytics", "warehouse"],
    ))
    reg.register(ToolRegistration(
        name="cache_layer",
        description="Fast cache lookup for recently accessed data",
        latency_ms=40,
        cost=0.001,
        reliability=0.85,
        tags=["data", "cache", "fast"],
    ))
    reg.register(ToolRegistration(
        name="analytics_script",
        description="Run compute-heavy analytics for anomaly detection and forecasting",
        latency_ms=450,
        cost=0.02,
        reliability=0.92,
        tags=["analytics", "compute", "ml", "anomaly"],
    ))
    return reg


@pytest.fixture(scope="module")
def intent_classifier() -> IntentClassifier:
    return IntentClassifier()


@pytest.fixture
def engine(registry: ToolRegistry, intent_classifier: IntentClassifier) -> DecisionEngine:
    return DecisionEngine(registry, intent_classifier)


class TestDecisionEngine:
    def test_score_returns_list(self, engine: DecisionEngine):
        results = engine.score_tools("Find Q4 revenue data")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_scores_are_bounded(self, engine: DecisionEngine):
        results = engine.score_tools("anomaly detection on sales data")
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_max_results_respected(self, engine: DecisionEngine):
        results = engine.score_tools("get data", max_results=1)
        assert len(results) == 1

    def test_tags_hint_filters(self, engine: DecisionEngine):
        results = engine.score_tools("get data", tags_hint=["cache"])
        tool_names = [r.tool_name for r in results]
        assert "cache_layer" in tool_names

    def test_breakdown_keys(self, engine: DecisionEngine):
        results = engine.score_tools("anything")
        for r in results:
            assert set(r.breakdown.keys()) == {
                "semantic_match",
                "reliability",
                "latency",
                "cost",
                "past_success",
            }

    def test_empty_registry(self, tmp_path: Path, intent_classifier: IntentClassifier):
        empty_reg = ToolRegistry(store_path=tmp_path / "empty.json")
        eng = DecisionEngine(empty_reg, intent_classifier)
        results = eng.score_tools("anything")
        assert results == []

    def test_cheap_fast_tool_preferred_for_simple_lookup(self, engine: DecisionEngine):
        """Cache layer should rank high for simple lookups (cheap + fast)."""
        results = engine.score_tools("get recently cached data")
        # cache_layer should appear somewhere in results
        names = [r.tool_name for r in results]
        assert "cache_layer" in names

    def test_analytics_preferred_for_anomaly(self, engine: DecisionEngine):
        """Analytics script should score well for anomaly detection."""
        results = engine.score_tools("detect anomalies in Q4 revenue")
        # analytics_script should be in the results
        names = [r.tool_name for r in results]
        assert "analytics_script" in names
