"""Unit tests for NonLLM context metric threshold consistency (issue #2777).

`NonLLMContextRecall` and `NonLLMContextPrecisionWithReference` share the same
default ``threshold`` (0.5) and the same ``NonLLMStringSimilarity`` distance
measure, so a per-context similarity score that lands exactly on the threshold
must be classified as relevant by both metrics. Previously recall used a strict
``>`` while precision used ``>=``, so a boundary score was relevant for
precision but not for recall.
"""

import numpy as np

from ragas.metrics._context_precision import NonLLMContextPrecisionWithReference
from ragas.metrics._context_recall import NonLLMContextRecall


class TestNonLLMThresholdConsistency:
    def test_recall_counts_score_at_threshold_as_relevant(self):
        """A similarity exactly at the threshold is relevant (>=), not dropped."""
        recall = NonLLMContextRecall(threshold=0.5)
        # Single context whose similarity equals the threshold.
        score = recall._compute_score([0.5])
        assert score == 1.0

    def test_recall_below_threshold_is_irrelevant(self):
        recall = NonLLMContextRecall(threshold=0.5)
        assert recall._compute_score([0.49]) == 0.0

    def test_recall_and_precision_agree_at_threshold(self):
        """Both metrics must classify a boundary score the same way."""
        threshold = 0.5
        recall = NonLLMContextRecall(threshold=threshold)
        precision = NonLLMContextPrecisionWithReference(threshold=threshold)

        boundary = [threshold]
        # recall: numerator/denom -> 1/1 == 1.0 when the boundary counts.
        recall_relevant = recall._compute_score(boundary) > 0
        # precision: average precision over a single relevant item == 1.0.
        precision_relevant = (
            precision._calculate_average_precision(
                [1 if s >= threshold else 0 for s in boundary]
            )
            > 0
        )

        assert recall_relevant == precision_relevant == True  # noqa: E712

    def test_recall_empty_returns_nan(self):
        recall = NonLLMContextRecall(threshold=0.5)
        assert np.isnan(recall._compute_score([]))
