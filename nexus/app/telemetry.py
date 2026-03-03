"""Telemetry — stores execution records and exposes aggregate metrics.

In-memory store backed by a JSON-lines file for persistence.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from nexus.models.schemas import ExecutionRecord, SystemMetrics, ToolMetrics, ToolRisk

_DEFAULT_LOG = Path(__file__).resolve().parent.parent / "data" / "telemetry.jsonl"


class TelemetryStore:
    """Thread-safe telemetry storage."""

    def __init__(self, log_path: Path | str | None = None) -> None:
        self._log_path = Path(log_path) if log_path else _DEFAULT_LOG
        self._lock = threading.Lock()
        self._records: list[ExecutionRecord] = []
        self._index: dict[str, int] = {}  # request_id → list index
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._log_path.exists():
            return
        with open(self._log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = ExecutionRecord(**json.loads(line))
                self._records.append(rec)
                self._index[rec.request_id] = len(self._records) - 1

    def _append_to_file(self, record: ExecutionRecord) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "a") as f:
            f.write(record.model_dump_json() + "\n")

    def _rewrite_file(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "w") as f:
            for rec in self._records:
                f.write(rec.model_dump_json() + "\n")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, record: ExecutionRecord) -> None:
        with self._lock:
            self._index[record.request_id] = len(self._records)
            self._records.append(record)
            self._append_to_file(record)

    def update(self, record: ExecutionRecord) -> None:
        with self._lock:
            idx = self._index.get(record.request_id)
            if idx is not None and idx < len(self._records):
                self._records[idx] = record
                self._rewrite_file()

    def get_by_request_id(self, request_id: str) -> Optional[ExecutionRecord]:
        idx = self._index.get(request_id)
        if idx is not None and idx < len(self._records):
            return self._records[idx]
        return None

    def list_all(self) -> list[ExecutionRecord]:
        return list(self._records)

    def list_for_tool(self, tool_name: str) -> list[ExecutionRecord]:
        return [r for r in self._records if r.tool_name == tool_name]

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._index.clear()
            if self._log_path.exists():
                self._log_path.unlink()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def tool_metrics(self, tool_name: str) -> ToolMetrics:
        records = self.list_for_tool(tool_name)
        total = len(records)
        successes = sum(1 for r in records if r.success)
        avg_lat = sum(r.latency_ms for r in records) / total if total else 0.0
        return ToolMetrics(
            tool_name=tool_name,
            total_calls=total,
            total_successes=successes,
            avg_latency_ms=round(avg_lat, 2),
            reputation=0.0,  # caller should enrich from registry
            success_rate=round(successes / total, 4) if total else 0.0,
        )

    def system_metrics(self, tool_names: list[str]) -> SystemMetrics:
        tools = [self.tool_metrics(n) for n in tool_names]
        return SystemMetrics(
            total_routes=len(self._records),
            tool_count=len(tool_names),
            tools=tools,
        )

    def tool_risk(self, tool_name: str, window: int = 20) -> ToolRisk:
        records = self.list_for_tool(tool_name)
        total = len(records)
        if total == 0:
            return ToolRisk(
                tool_name=tool_name,
                risk_score=0.0,
                current_success_rate=1.0,
                success_rate_drop=0.0,
                recent_failure_rate=0.0,
                latency_drift=0.0,
                recent_avg_latency_ms=0.0,
                baseline_avg_latency_ms=0.0,
                sample_size=0,
            )

        successes = sum(1 for r in records if r.success)
        current_success_rate = successes / total

        slice_window = max(1, window)
        recent = records[-slice_window:]
        previous = records[-(slice_window * 2):-slice_window]

        recent_success = sum(1 for r in recent if r.success) / len(recent)
        previous_success = (
            sum(1 for r in previous if r.success) / len(previous)
            if previous
            else current_success_rate
        )
        success_rate_drop = max(previous_success - recent_success, 0.0)

        recent_failure_rate = 1.0 - recent_success

        recent_avg_latency = sum(r.latency_ms for r in recent) / len(recent)
        baseline_avg_latency = sum(r.latency_ms for r in records) / len(records)
        latency_drift = (
            max((recent_avg_latency - baseline_avg_latency) / baseline_avg_latency, 0.0)
            if baseline_avg_latency > 0
            else 0.0
        )

        risk_score = (
            0.40 * (1.0 - current_success_rate)
            + 0.25 * success_rate_drop
            + 0.20 * min(latency_drift, 1.0)
            + 0.15 * recent_failure_rate
        )
        risk_score = max(0.0, min(1.0, risk_score))

        return ToolRisk(
            tool_name=tool_name,
            risk_score=round(risk_score, 4),
            current_success_rate=round(current_success_rate, 4),
            success_rate_drop=round(success_rate_drop, 4),
            recent_failure_rate=round(recent_failure_rate, 4),
            latency_drift=round(latency_drift, 4),
            recent_avg_latency_ms=round(recent_avg_latency, 2),
            baseline_avg_latency_ms=round(baseline_avg_latency, 2),
            sample_size=total,
        )

    def top_risks(self, tool_names: list[str], window: int = 20, limit: int = 5) -> list[ToolRisk]:
        risks = [self.tool_risk(tool_name=name, window=window) for name in tool_names]
        risks.sort(key=lambda item: (-item.risk_score, -item.sample_size, item.tool_name))
        return risks[: max(1, limit)]
