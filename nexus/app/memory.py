"""Memory / Learning Loop — updates tool reputation based on execution history.

Implements a simple reinforcement mechanism:
  - Success → reputation nudged up
  - Failure → reputation nudged down
  - Decays toward 0.5 over time (mean-reverting)

Can be run periodically or triggered after each execution.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from nexus.app.registry import ToolRegistry
from nexus.app.telemetry import TelemetryStore

logger = logging.getLogger(__name__)


@dataclass
class LearningConfig:
    """Tuneable knobs for the reputation update."""

    success_boost: float = 0.02
    failure_penalty: float = 0.05
    decay_rate: float = 0.001  # per-update drift toward baseline
    baseline: float = 0.5
    min_reputation: float = 0.05
    max_reputation: float = 0.99


class MemoryManager:
    """Adjusts tool reputation scores based on execution telemetry."""

    def __init__(
        self,
        registry: ToolRegistry,
        telemetry: TelemetryStore,
        config: LearningConfig | None = None,
    ) -> None:
        self.registry = registry
        self.telemetry = telemetry
        self.config = config or LearningConfig()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update_reputation(self, tool_name: str, success: bool) -> float:
        """Apply a single reinforcement step and return the new reputation."""
        tool = self.registry.get(tool_name)
        if tool is None:
            logger.warning("MemoryManager: unknown tool '%s'", tool_name)
            return 0.0

        cfg = self.config
        rep = tool.reputation

        # Apply reward / penalty
        if success:
            rep += cfg.success_boost
        else:
            rep -= cfg.failure_penalty

        # Mean-reverting decay toward baseline
        rep += cfg.decay_rate * (cfg.baseline - rep)

        # Clamp
        rep = max(cfg.min_reputation, min(cfg.max_reputation, rep))

        self.registry.update_tool(tool_name, reputation=round(rep, 6))
        logger.info(
            "Reputation updated: %s  %.4f → %.4f  (success=%s)",
            tool_name,
            tool.reputation,
            rep,
            success,
        )
        return rep

    def recalculate_all(self) -> dict[str, float]:
        """Recompute reputation from full telemetry history (cold start / reset).

        Uses a weighted moving average over all historical execution records.
        """
        results: dict[str, float] = {}
        for tool in self.registry.list_all():
            records = self.telemetry.list_for_tool(tool.name)
            if not records:
                results[tool.name] = tool.reputation
                continue

            # Exponential weighted average (newer records weigh more)
            total_weight = 0.0
            weighted_success = 0.0
            for i, rec in enumerate(records):
                w = math.exp(0.03 * i)  # exponential recency weight
                total_weight += w
                weighted_success += w * (1.0 if rec.success else 0.0)

            rep = weighted_success / total_weight if total_weight else self.config.baseline
            rep = max(self.config.min_reputation, min(self.config.max_reputation, rep))
            self.registry.update_tool(tool.name, reputation=round(rep, 6))
            results[tool.name] = rep

        return results
