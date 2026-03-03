"""Tests for Telemetry store."""

from datetime import datetime
from pathlib import Path

import pytest

from nexus.app.telemetry import TelemetryStore
from nexus.models.schemas import ExecutionRecord


def _make_record(request_id: str = "req-1", tool_name: str = "t1", success: bool = True) -> ExecutionRecord:
    return ExecutionRecord(
        request_id=request_id,
        tool_name=tool_name,
        tool_id="id-1",
        query="test query",
        success=success,
        latency_ms=42.0,
    )


@pytest.fixture
def store(tmp_path: Path) -> TelemetryStore:
    return TelemetryStore(log_path=tmp_path / "tel.jsonl")


class TestTelemetryStore:
    def test_add_and_retrieve(self, store: TelemetryStore):
        rec = _make_record()
        store.add(rec)
        assert store.get_by_request_id("req-1") is not None

    def test_list_all(self, store: TelemetryStore):
        store.add(_make_record("r1"))
        store.add(_make_record("r2"))
        assert len(store.list_all()) == 2

    def test_list_for_tool(self, store: TelemetryStore):
        store.add(_make_record("r1", tool_name="a"))
        store.add(_make_record("r2", tool_name="b"))
        store.add(_make_record("r3", tool_name="a"))
        assert len(store.list_for_tool("a")) == 2

    def test_update(self, store: TelemetryStore):
        rec = _make_record("upd")
        store.add(rec)
        updated = rec.model_copy(update={"success": False, "error_message": "oops"})
        store.update(updated)
        result = store.get_by_request_id("upd")
        assert result is not None
        assert result.success is False
        assert result.error_message == "oops"

    def test_tool_metrics(self, store: TelemetryStore):
        store.add(_make_record("r1", tool_name="x", success=True))
        store.add(_make_record("r2", tool_name="x", success=True))
        store.add(_make_record("r3", tool_name="x", success=False))
        m = store.tool_metrics("x")
        assert m.total_calls == 3
        assert m.total_successes == 2
        assert abs(m.success_rate - 2 / 3) < 0.01

    def test_system_metrics(self, store: TelemetryStore):
        store.add(_make_record("r1", tool_name="a"))
        store.add(_make_record("r2", tool_name="b"))
        sm = store.system_metrics(["a", "b"])
        assert sm.total_routes == 2
        assert sm.tool_count == 2

    def test_persistence(self, tmp_path: Path):
        path = tmp_path / "persist.jsonl"
        s1 = TelemetryStore(log_path=path)
        s1.add(_make_record("p1"))
        del s1

        s2 = TelemetryStore(log_path=path)
        assert s2.get_by_request_id("p1") is not None

    def test_clear(self, store: TelemetryStore):
        store.add(_make_record("c1"))
        store.clear()
        assert len(store.list_all()) == 0
