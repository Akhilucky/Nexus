"""Tests for the Memory / Learning Loop."""

from pathlib import Path

import pytest

from nexus.app.memory import LearningConfig, MemoryManager
from nexus.app.registry import ToolRegistry
from nexus.app.telemetry import TelemetryStore
from nexus.models.schemas import ExecutionRecord, ToolRegistration


@pytest.fixture
def setup(tmp_path: Path):
    registry = ToolRegistry(store_path=tmp_path / "tools.json")
    registry.register(ToolRegistration(name="tool_a", description="desc"))
    telemetry = TelemetryStore(log_path=tmp_path / "tel.jsonl")
    memory = MemoryManager(registry, telemetry)
    return memory, registry, telemetry


class TestMemoryManager:
    def test_success_boosts_reputation(self, setup):
        memory, registry, _ = setup
        before = registry.get("tool_a").reputation
        memory.update_reputation("tool_a", success=True)
        after = registry.get("tool_a").reputation
        assert after > before

    def test_failure_lowers_reputation(self, setup):
        memory, registry, _ = setup
        before = registry.get("tool_a").reputation
        memory.update_reputation("tool_a", success=False)
        after = registry.get("tool_a").reputation
        assert after < before

    def test_reputation_clamped(self, setup):
        memory, registry, _ = setup
        # Many failures should not go below min
        for _ in range(100):
            memory.update_reputation("tool_a", success=False)
        rep = registry.get("tool_a").reputation
        assert rep >= LearningConfig().min_reputation

    def test_reputation_clamped_high(self, setup):
        memory, registry, _ = setup
        for _ in range(100):
            memory.update_reputation("tool_a", success=True)
        rep = registry.get("tool_a").reputation
        assert rep <= LearningConfig().max_reputation

    def test_recalculate_all(self, setup):
        memory, registry, telemetry = setup
        # Add some history
        for i in range(5):
            telemetry.add(ExecutionRecord(
                request_id=f"r{i}",
                tool_name="tool_a",
                tool_id="id",
                query="q",
                success=(i % 2 == 0),
                latency_ms=10.0,
            ))
        results = memory.recalculate_all()
        assert "tool_a" in results
        assert 0.0 <= results["tool_a"] <= 1.0

    def test_unknown_tool_returns_zero(self, setup):
        memory, _, _ = setup
        rep = memory.update_reputation("nonexistent", success=True)
        assert rep == 0.0
