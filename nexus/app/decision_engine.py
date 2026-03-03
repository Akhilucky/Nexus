"""Decision Engine — scores and ranks tools for a given query.

Scoring formula (from spec):
    score =
        semantic_match * 0.40
      + reliability     * 0.20
      + latency_weight  * 0.15
      + cost_weight     * 0.15
      + past_success    * 0.10

All sub-scores are normalised to [0, 1] before weighting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from nexus.app.intent import IntentClassifier
from nexus.app.registry import ToolRegistry
from nexus.models.schemas import ScoredTool, Tool

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    semantic_match: float = 0.40
    reliability: float = 0.20
    latency: float = 0.15
    cost: float = 0.15
    past_success: float = 0.10

    def as_dict(self) -> dict[str, float]:
        return {
            "semantic_match": self.semantic_match,
            "reliability": self.reliability,
            "latency": self.latency,
            "cost": self.cost,
            "past_success": self.past_success,
        }


class DecisionEngine:
    """Rank registered tools for a given user query."""

    def __init__(
        self,
        registry: ToolRegistry,
        intent_classifier: IntentClassifier,
        weights: ScoringWeights | None = None,
    ) -> None:
        self.registry = registry
        self.intent = intent_classifier
        self.weights = weights or ScoringWeights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_tools(
        self,
        query: str,
        tags_hint: list[str] | None = None,
        max_results: int = 3,
        candidate_tools: list[Tool] | None = None,
    ) -> list[ScoredTool]:
        """Return a ranked list of ScoredTool objects for *query*."""
        tools = candidate_tools if candidate_tools is not None else self._candidate_tools(tags_hint)
        if not tools:
            return []

        query_embedding = self.intent.encode(query)  # may be None

        scored: list[ScoredTool] = []
        for tool in tools:
            breakdown = self._compute_breakdown(query, query_embedding, tool, tools)
            total = sum(
                self.weights.as_dict()[k] * v for k, v in breakdown.items()
            )
            scored.append(
                ScoredTool(
                    tool_name=tool.name,
                    tool_id=tool.id,
                    score=round(total, 4),
                    breakdown={k: round(v, 4) for k, v in breakdown.items()},
                )
            )

        scored.sort(
            key=lambda s: (
                -s.score,
                self.registry.get(s.tool_name).latency_ms if self.registry.get(s.tool_name) else float("inf"),
                self.registry.get(s.tool_name).cost if self.registry.get(s.tool_name) else float("inf"),
                s.tool_name,
            )
        )
        return scored[:max_results]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _candidate_tools(self, tags_hint: list[str] | None) -> list[Tool]:
        """Filter tools by tags if a hint is provided."""
        all_tools = self.registry.list_all()
        if not tags_hint:
            return all_tools
        tag_set = set(t.lower() for t in tags_hint)
        return [
            t for t in all_tools
            if tag_set & set(tag.lower() for tag in t.tags)
        ]

    def _compute_breakdown(
        self,
        query: str,
        query_embedding: np.ndarray | None,
        tool: Tool,
        all_tools: list[Tool],
    ) -> dict[str, float]:
        """Compute normalised per-factor scores for a single tool."""
        return {
            "semantic_match": self._semantic_score(query, query_embedding, tool),
            "reliability": tool.reliability,
            "latency": self._latency_score(tool, all_tools),
            "cost": self._cost_score(tool, all_tools),
            "past_success": self._past_success_score(tool),
        }

    # --- sub-scores ---------------------------------------------------

    def _semantic_score(
        self, query: str, query_embedding: np.ndarray | None, tool: Tool
    ) -> float:
        """Cosine similarity between query and tool description embeddings."""
        if query_embedding is None:
            # Keyword fallback: fraction of query words found in description
            q_words = set(query.lower().split())
            desc_words = set(tool.description.lower().split())
            tag_words = set(t.lower() for t in tool.tags)
            overlap = q_words & (desc_words | tag_words)
            return min(len(overlap) / max(len(q_words), 1), 1.0)

        tool_embedding = self._get_tool_embedding(tool)
        if tool_embedding is None:
            return 0.0
        sim = float(np.dot(query_embedding, tool_embedding))
        return max(0.0, sim)

    def _get_tool_embedding(self, tool: Tool) -> np.ndarray | None:
        if tool.embedding is not None:
            return np.array(tool.embedding)
        emb = self.intent.encode(f"{tool.name}: {tool.description} {' '.join(tool.tags)}")
        if emb is not None:
            self.registry.update_tool(tool.name, embedding=emb.tolist())
        return emb

    @staticmethod
    def _latency_score(tool: Tool, all_tools: list[Tool]) -> float:
        """Lower latency → higher score (inverted normalisation)."""
        latencies = [t.latency_ms for t in all_tools]
        min_lat = min(latencies)
        max_lat = max(latencies)
        if max_lat == min_lat:
            return 1.0
        score = 1.0 - ((tool.latency_ms - min_lat) / (max_lat - min_lat))
        return max(0.0, min(1.0, score))

    @staticmethod
    def _cost_score(tool: Tool, all_tools: list[Tool]) -> float:
        """Lower cost → higher score."""
        costs = [t.cost for t in all_tools]
        min_cost = min(costs)
        max_cost = max(costs)
        if max_cost == min_cost:
            return 1.0
        score = 1.0 - ((tool.cost - min_cost) / (max_cost - min_cost))
        return max(0.0, min(1.0, score))

    @staticmethod
    def _past_success_score(tool: Tool) -> float:
        """Reputation is already 0-1; combine with empirical success rate."""
        if tool.total_calls == 0:
            return tool.reputation
        empirical = tool.total_successes / tool.total_calls
        # Blend reputation with empirical data (more data → more empirical)
        alpha = min(tool.total_calls / 100, 1.0)
        return (1 - alpha) * tool.reputation + alpha * empirical
