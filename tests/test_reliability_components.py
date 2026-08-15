import numpy as np
import pytest

from aegis.ood.zscore import ZScoreOODDetector
from aegis.scoring.aggregator import ScoreAggregator, ScoreInputs
from aegis.scoring.weighting import ScoreWeights
from aegis.scoring.trust import TrustScorer
from aegis.uncertainty.confidence import ConfidenceEstimator
from aegis.uncertainty.entropy import EntropyEstimator


def test_confidence_uses_maximum_probability():
    probabilities = np.array([[0.1, 0.9], [0.6, 0.4]])
    result = ConfidenceEstimator().compute(probabilities)
    np.testing.assert_allclose(result, [0.9, 0.6])


def test_entropy_is_normalized():
    estimator = EntropyEstimator()
    result = estimator.compute(np.array([[1.0, 0.0], [0.5, 0.5]]))
    np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-12)


def test_ood_score_is_raw_zscore_and_threshold_is_explicit():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    detector = ZScoreOODDetector(threshold=2.0).fit(X)

    scores = detector.score(np.array([[0.0, 0.0], [20.0, 20.0]]))

    assert scores[1] > 2.0
    assert detector.predict(np.array([[20.0, 20.0]])).tolist() == [True]


def test_ood_rejects_feature_count_mismatch():
    detector = ZScoreOODDetector().fit(np.zeros((5, 3)))
    with pytest.raises(ValueError, match="features"):
        detector.score(np.zeros((1, 2)))


def test_score_weights_normalize_and_reject_invalid_values():
    weights = ScoreWeights(0.5, 0.3, 0.2, 0.0).normalize()
    assert weights.confidence + weights.uncertainty + weights.ood + weights.drift == pytest.approx(1.0)

    with pytest.raises(ValueError):
        ScoreWeights(0.0, 0.0, 0.0, 0.0).normalize()


def test_trust_scorer_returns_typed_decision():
    scorer = TrustScorer(ScoreAggregator())
    result = scorer.compute(
        confidence=0.95,
        entropy=0.05,
        ood_score=0.0,
        is_ood=False,
    )

    assert result.trust_score > 0.85
    assert result.risk_level.value == "LOW"
