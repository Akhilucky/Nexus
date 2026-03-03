"""Intent Classifier — lightweight embedding-based intent detection.

Uses sentence-transformers for semantic similarity between the user query
and predefined intent descriptions.  Falls back to keyword matching when
the model is unavailable.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from nexus.models.schemas import IntentCategory

logger = logging.getLogger(__name__)

# Canonical descriptions for each intent (used for embedding similarity)
_INTENT_DESCRIPTIONS: dict[IntentCategory, str] = {
    IntentCategory.retrieval: "Find, search, look up, fetch, get, retrieve data or information from a source",
    IntentCategory.computation: "Calculate, compute, analyze numbers, run algorithms, process data mathematically",
    IntentCategory.visualization: "Plot, chart, graph, visualize, display, render visual representation of data",
    IntentCategory.automation: "Automate, schedule, trigger, run workflow, execute pipeline, deploy",
    IntentCategory.summarization: "Summarize, condense, extract key points, distill, abstract, shorten text",
}

# Simple keyword fallback
_KEYWORD_MAP: dict[IntentCategory, list[str]] = {
    IntentCategory.retrieval: ["find", "search", "get", "fetch", "look up", "retrieve", "query", "list", "show"],
    IntentCategory.computation: ["calculate", "compute", "analyze", "average", "sum", "count", "predict", "anomal"],
    IntentCategory.visualization: ["plot", "chart", "graph", "visualize", "display", "render", "draw", "dashboard"],
    IntentCategory.automation: ["automate", "schedule", "trigger", "deploy", "run", "execute", "pipeline", "workflow"],
    IntentCategory.summarization: ["summarize", "summary", "condense", "key points", "distill", "shorten", "tldr"],
}


class IntentClassifier:
    """Classify user queries into intent categories using embeddings or keywords."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = None
        self._model_name = model_name
        self._intent_embeddings: dict[IntentCategory, np.ndarray] | None = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            self._precompute_intent_embeddings()
            logger.info("IntentClassifier: sentence-transformers model loaded (%s)", self._model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("IntentClassifier: could not load model (%s), using keyword fallback", exc)
            self._model = None

    def _precompute_intent_embeddings(self) -> None:
        if self._model is None:
            return
        self._intent_embeddings = {}
        for intent, desc in _INTENT_DESCRIPTIONS.items():
            self._intent_embeddings[intent] = self._model.encode(desc, normalize_embeddings=True)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def encode(self, text: str) -> Optional[np.ndarray]:
        """Encode text into a normalized embedding vector. Returns None if model unavailable."""
        if self._model is None:
            return None
        return self._model.encode(text, normalize_embeddings=True)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, query: str) -> tuple[IntentCategory, dict[str, float]]:
        """Return (best_intent, {intent: score}) for a query."""
        if self._model is not None and self._intent_embeddings is not None:
            return self._classify_embedding(query)
        return self._classify_keywords(query)

    def _classify_embedding(self, query: str) -> tuple[IntentCategory, dict[str, float]]:
        q_emb = self._model.encode(query, normalize_embeddings=True)  # type: ignore[union-attr]
        scores: dict[str, float] = {}
        for intent, i_emb in self._intent_embeddings.items():  # type: ignore[union-attr]
            sim = float(np.dot(q_emb, i_emb))
            scores[intent.value] = max(0.0, sim)  # clamp negatives

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_intent = IntentCategory(best)
        if scores[best] < 0.15:
            best_intent = IntentCategory.unknown
        return best_intent, scores

    def _classify_keywords(self, query: str) -> tuple[IntentCategory, dict[str, float]]:
        q_lower = query.lower()
        scores: dict[str, float] = {}
        for intent, keywords in _KEYWORD_MAP.items():
            hits = sum(1 for kw in keywords if kw in q_lower)
            scores[intent.value] = hits / len(keywords)

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_intent = IntentCategory(best) if scores[best] > 0 else IntentCategory.unknown
        return best_intent, scores
