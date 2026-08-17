import math

import pytest

from aegis.core.enums import Recommendation, RiskLevel
from aegis.core.exceptions import ConfigurationError, ValidationError
from aegis.scoring.aggregator import ScoreAggregator, ScoreInputs
from aegis.scoring.trust import TrustScorer
from aegis.scoring.weighting import ScoreWeights


def test_default_weights_sum_to_one():
    weights = ScoreWeights().normalize()

    total = (
        weights.confidence
        + weights.uncertainty
        + weights.ood
        + weights.drift
    )

    assert total == pytest.approx(1.0)


def test_custom_weights_are_normalized():
    weights = ScoreWeights(
        confidence=5.0,
        uncertainty=3.0,
        ood=2.0,
        drift=0.0,
    ).normalize()

    assert weights.confidence == pytest.approx(0.5)
    assert weights.uncertainty == pytest.approx(0.3)
    assert weights.ood == pytest.approx(0.2)
    assert weights.drift == pytest.approx(0.0)


@pytest.mark.parametrize(
    "weights",
    [
        ScoreWeights(-1.0, 0.5, 0.5, 0.0),
        ScoreWeights(0.5, -1.0, 0.5, 0.0),
        ScoreWeights(0.5, 0.5, -1.0, 0.0),
        ScoreWeights(0.5, 0.5, 0.0, -1.0),
    ],
)
def test_negative_weights_are_rejected(weights):
    with pytest.raises(
        ConfigurationError,
        match="cannot be negative",
    ):
        weights.normalize()


@pytest.mark.parametrize(
    "weights",
    [
        ScoreWeights(float("nan"), 0.3, 0.7, 0.0),
        ScoreWeights(0.5, float("inf"), 0.5, 0.0),
        ScoreWeights(0.5, 0.5, float("-inf"), 0.0),
    ],
)
def test_non_finite_weights_are_rejected(weights):
    with pytest.raises(
        ConfigurationError,
        match="finite",
    ):
        weights.normalize()


def test_all_zero_weights_are_rejected():
    with pytest.raises(
        ConfigurationError,
        match="greater than zero",
    ):
        ScoreWeights(
            0.0,
            0.0,
            0.0,
            0.0,
        ).normalize()


def test_drift_is_disabled_by_default():
    weights = ScoreWeights().normalize()

    assert weights.drift == pytest.approx(0.0)
    assert weights.is_drift_enabled() is False


def test_drift_can_be_explicitly_enabled():
    weights = ScoreWeights(
        confidence=0.4,
        uncertainty=0.2,
        ood=0.2,
        drift=0.2,
    ).normalize()

    assert weights.is_drift_enabled() is True
    assert weights.drift == pytest.approx(0.2)


def test_weights_as_dict():
    weights = ScoreWeights(
        0.5,
        0.3,
        0.2,
        0.0,
    ).normalize()

    result = weights.as_dict()

    assert result == {
        "confidence": pytest.approx(0.5),
        "uncertainty": pytest.approx(0.3),
        "ood": pytest.approx(0.2),
        "drift": pytest.approx(0.0),
    }


def test_perfect_reliability_produces_maximum_trust():
    aggregator = ScoreAggregator(
        ScoreWeights(
            0.5,
            0.3,
            0.2,
            0.0,
        )
    )

    result = aggregator.aggregate(
        ScoreInputs(
            confidence=1.0,
            uncertainty=0.0,
            ood_risk=0.0,
            drift=0.0,
        )
    )

    assert result == pytest.approx(1.0)


def test_worst_reliability_produces_minimum_trust():
    aggregator = ScoreAggregator(
        ScoreWeights(
            0.5,
            0.3,
            0.2,
            0.0,
        )
    )

    result = aggregator.aggregate(
        ScoreInputs(
            confidence=0.0,
            uncertainty=1.0,
            ood_risk=1.0,
            drift=0.0,
        )
    )

    assert result == pytest.approx(0.0)


def test_score_is_weighted_correctly():
    aggregator = ScoreAggregator(
        ScoreWeights(
            0.5,
            0.3,
            0.2,
            0.0,
        )
    )

    result = aggregator.aggregate(
        ScoreInputs(
            confidence=0.8,
            uncertainty=0.2,
            ood_risk=0.1,
            drift=0.0,
        )
    )

    expected = (
        0.5 * 0.8
        + 0.3 * (1.0 - 0.2)
        + 0.2 * (1.0 - 0.1)
    )

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("uncertainty", -0.1),
        ("uncertainty", 1.1),
        ("ood_risk", -0.1),
        ("ood_risk", 1.1),
        ("drift", -0.1),
        ("drift", 1.1),
    ],
)
def test_score_inputs_must_be_between_zero_and_one(field, value):
    values = {
        "confidence": 0.5,
        "uncertainty": 0.5,
        "ood_risk": 0.5,
        "drift": 0.0,
    }

    values[field] = value

    inputs = ScoreInputs(**values)

    with pytest.raises(
        ValidationError,
        match=field,
    ):
        ScoreAggregator().aggregate(inputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("confidence", float("nan")),
        ("confidence", float("inf")),
        ("uncertainty", float("nan")),
        ("ood_risk", float("inf")),
        ("drift", float("nan")),
    ],
)
def test_non_finite_score_inputs_are_rejected(field, value):
    values = {
        "confidence": 0.5,
        "uncertainty": 0.5,
        "ood_risk": 0.5,
        "drift": 0.0,
    }

    values[field] = value

    inputs = ScoreInputs(**values)

    with pytest.raises(
        ValidationError,
        match="finite",
    ):
        ScoreAggregator().aggregate(inputs)


def test_zero_uncertainty_is_best():
    aggregator = ScoreAggregator()

    low_uncertainty = aggregator.aggregate(
        ScoreInputs(
            confidence=0.8,
            uncertainty=0.0,
            ood_risk=0.0,
        )
    )

    high_uncertainty = aggregator.aggregate(
        ScoreInputs(
            confidence=0.8,
            uncertainty=1.0,
            ood_risk=0.0,
        )
    )

    assert low_uncertainty > high_uncertainty


def test_zero_ood_risk_is_best():
    aggregator = ScoreAggregator()

    safe = aggregator.aggregate(
        ScoreInputs(
            confidence=0.8,
            uncertainty=0.1,
            ood_risk=0.0,
        )
    )

    risky = aggregator.aggregate(
        ScoreInputs(
            confidence=0.8,
            uncertainty=0.1,
            ood_risk=1.0,
        )
    )

    assert safe > risky


def test_ood_risk_reduces_trust():
    aggregator = ScoreAggregator()

    result = aggregator.aggregate(
        ScoreInputs(
            confidence=0.9,
            uncertainty=0.1,
            ood_risk=1.0,
        )
    )

    baseline = aggregator.aggregate(
        ScoreInputs(
            confidence=0.9,
            uncertainty=0.1,
            ood_risk=0.0,
        )
    )

    assert result < baseline


def test_drift_reduces_trust_when_enabled():
    aggregator = ScoreAggregator(
        ScoreWeights(
            confidence=0.4,
            uncertainty=0.2,
            ood=0.2,
            drift=0.2,
        )
    )

    no_drift = aggregator.aggregate(
        ScoreInputs(
            confidence=0.8,
            uncertainty=0.1,
            ood_risk=0.0,
            drift=0.0,
        )
    )

    high_drift = aggregator.aggregate(
        ScoreInputs(
            confidence=0.8,
            uncertainty=0.1,
            ood_risk=0.0,
            drift=1.0,
        )
    )

    assert high_drift < no_drift


def test_trust_score_is_always_bounded():
    aggregator = ScoreAggregator()

    result = aggregator.aggregate(
        ScoreInputs(
            confidence=1.0,
            uncertainty=0.0,
            ood_risk=0.0,
        )
    )

    assert 0.0 <= result <= 1.0


def test_low_trust_produces_high_risk():
    scorer = TrustScorer()

    result = scorer.compute(
        confidence=0.1,
        entropy=0.9,
        ood_score=0.9,
        is_ood=False,
    )

    assert result.risk_level == RiskLevel.HIGH
    assert result.recommendation == Recommendation.REJECT


def test_medium_trust_requires_human_review():
    scorer = TrustScorer()

    result = scorer.compute(
        confidence=0.6,
        entropy=0.4,
        ood_score=0.0,
        is_ood=False,
    )

    assert 0.60 <= result.trust_score < 0.85
    assert result.risk_level == RiskLevel.MEDIUM
    assert result.recommendation == Recommendation.HUMAN_REVIEW


def test_high_trust_auto_approves():
    scorer = TrustScorer()

    result = scorer.compute(
        confidence=0.95,
        entropy=0.05,
        ood_score=0.0,
        is_ood=False,
    )

    assert result.trust_score >= 0.85
    assert result.risk_level == RiskLevel.LOW
    assert result.recommendation == Recommendation.AUTO_APPROVE


def test_ood_boolean_is_hard_risk_signal():
    scorer = TrustScorer()

    normal = scorer.compute(
        confidence=0.95,
        entropy=0.05,
        ood_score=0.0,
        is_ood=False,
    )

    ood = scorer.compute(
        confidence=0.95,
        entropy=0.05,
        ood_score=0.0,
        is_ood=True,
    )

    assert ood.trust_score < normal.trust_score


def test_trust_result_is_typed():
    scorer = TrustScorer()

    result = scorer.compute(
        confidence=0.95,
        entropy=0.05,
        ood_score=0.0,
        is_ood=False,
    )

    assert isinstance(result.trust_score, float)
    assert isinstance(result.risk_level, RiskLevel)
    assert isinstance(result.recommendation, Recommendation)


def test_trust_score_is_deterministic():
    scorer = TrustScorer()

    inputs = {
        "confidence": 0.82,
        "entropy": 0.18,
        "ood_score": 0.10,
        "is_ood": False,
    }

    first = scorer.compute(**inputs)
    second = scorer.compute(**inputs)

    assert first == second


def test_score_inputs_are_immutable():
    inputs = ScoreInputs(
        confidence=0.8,
        uncertainty=0.2,
        ood_risk=0.1,
        drift=0.0,
    )

    with pytest.raises(AttributeError):
        inputs.confidence = 0.5


def test_weights_are_immutable():
    weights = ScoreWeights()

    with pytest.raises(AttributeError):
        weights.confidence = 0.2


def test_custom_aggregator_is_used():
    weights = ScoreWeights(
        confidence=1.0,
        uncertainty=0.0,
        ood=0.0,
        drift=0.0,
    )

    scorer = TrustScorer(
        ScoreAggregator(weights)
    )

    result = scorer.compute(
        confidence=0.73,
        entropy=0.99,
        ood_score=0.99,
        is_ood=False,
    )

    assert result.trust_score == pytest.approx(0.73)


def test_nan_trust_inputs_fail():
    scorer = TrustScorer()

    with pytest.raises(ValidationError):
        scorer.compute(
            confidence=math.nan,
            entropy=0.1,
            ood_score=0.1,
            is_ood=False,
        )


def test_infinite_trust_inputs_fail():
    scorer = TrustScorer()

    with pytest.raises(ValidationError):
        scorer.compute(
            confidence=math.inf,
            entropy=0.1,
            ood_score=0.1,
            is_ood=False,
        )