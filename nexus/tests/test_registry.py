"""Tests for the Tool Registry."""

import json
import tempfile
from pathlib import Path

import pytest

from nexus.app.registry import ToolRegistry
from nexus.models.schemas import ToolRegistration


@pytest.fixture
def tmp_store(tmp_path: Path) -> Path:
    return tmp_path / "tools.json"


@pytest.fixture
def registry(tmp_store: Path) -> ToolRegistry:
    return ToolRegistry(store_path=tmp_store)


class TestToolRegistry:
    def test_register_new_tool(self, registry: ToolRegistry):
        payload = ToolRegistration(
            name="test_tool",
            description="A test tool",
            latency_ms=100,
            cost=0.01,
            reliability=0.95,
            tags=["test"],
        )
        tool = registry.register(payload)
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.reputation == 0.5

    def test_register_duplicate_updates(self, registry: ToolRegistry):
        p1 = ToolRegistration(name="dup", description="v1", latency_ms=100)
        p2 = ToolRegistration(name="dup", description="v2", latency_ms=200)
        registry.register(p1)
        updated = registry.register(p2)
        assert updated.description == "v2"
        assert updated.latency_ms == 200
        assert len(registry.list_all()) == 1

    def test_get(self, registry: ToolRegistry):
        registry.register(ToolRegistration(name="x", description="desc"))
        assert registry.get("x") is not None
        assert registry.get("nonexistent") is None

    def test_list_all(self, registry: ToolRegistry):
        for i in range(3):
            registry.register(ToolRegistration(name=f"tool_{i}", description=f"desc {i}"))
        assert len(registry.list_all()) == 3

    def test_remove(self, registry: ToolRegistry):
        registry.register(ToolRegistration(name="rm_me", description="bye"))
        assert registry.remove("rm_me") is True
        assert registry.get("rm_me") is None
        assert registry.remove("rm_me") is False

    def test_update_tool(self, registry: ToolRegistry):
        registry.register(ToolRegistration(name="up", description="d"))
        updated = registry.update_tool("up", reputation=0.9, total_calls=10)
        assert updated is not None
        assert updated.reputation == 0.9
        assert updated.total_calls == 10

    def test_persistence(self, tmp_store: Path):
        reg1 = ToolRegistry(store_path=tmp_store)
        reg1.register(ToolRegistration(name="persist", description="test"))
        del reg1

        reg2 = ToolRegistry(store_path=tmp_store)
        assert reg2.get("persist") is not None
        assert reg2.get("persist").description == "test"

    def test_clear(self, registry: ToolRegistry):
        registry.register(ToolRegistration(name="a", description="d"))
        registry.clear()
        assert len(registry.list_all()) == 0
