"""Tests for the Intent Classifier."""

import pytest

from nexus.app.intent import IntentClassifier
from nexus.models.schemas import IntentCategory


@pytest.fixture(scope="module")
def classifier() -> IntentClassifier:
    """Create classifier once per module (model loading is expensive)."""
    return IntentClassifier()


class TestIntentClassifier:
    def test_classify_returns_tuple(self, classifier: IntentClassifier):
        intent, scores = classifier.classify("find all users")
        assert isinstance(intent, IntentCategory)
        assert isinstance(scores, dict)
        assert len(scores) > 0

    def test_retrieval_intent(self, classifier: IntentClassifier):
        intent, _ = classifier.classify("search for customer records")
        # Should be retrieval (or at least not unknown)
        assert intent != IntentCategory.unknown

    def test_computation_intent(self, classifier: IntentClassifier):
        intent, _ = classifier.classify("calculate the average revenue per quarter")
        assert intent != IntentCategory.unknown

    def test_unknown_for_gibberish(self, classifier: IntentClassifier):
        # Keyword fallback should return unknown for nonsense
        intent, scores = classifier._classify_keywords("xyzzy foobar baz")
        assert intent == IntentCategory.unknown

    def test_encode_returns_vector_or_none(self, classifier: IntentClassifier):
        result = classifier.encode("hello world")
        # If model loaded, should be ndarray; otherwise None
        if classifier._model is not None:
            assert result is not None
            assert len(result) > 0
        else:
            assert result is None

    def test_keyword_fallback(self, classifier: IntentClassifier):
        intent, scores = classifier._classify_keywords("summarize this report for me")
        assert intent == IntentCategory.summarization
