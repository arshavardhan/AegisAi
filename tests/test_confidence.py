import numpy as np
import pytest

from aegis.core.exceptions import ValidationError
from aegis.uncertainty.confidence import ConfidenceEstimator


@pytest.fixture
def estimator():
    return ConfidenceEstimator()


def test_name(estimator):
    assert estimator.name == "Maximum Probability Confidence"


def test_binary_confidence(estimator):
    probabilities = np.array(
        [
            [0.10, 0.90],
            [0.60, 0.40],
        ]
    )

    result = estimator.compute(probabilities)

    np.testing.assert_allclose(
        result,
        [0.90, 0.60],
    )


def test_multiclass_confidence(estimator):
    probabilities = np.array(
        [
            [0.10, 0.20, 0.70],
            [0.80, 0.10, 0.10],
        ]
    )

    result = estimator.compute(probabilities)

    np.testing.assert_allclose(
        result,
        [0.70, 0.80],
    )


def test_perfect_confidence(estimator):
    probabilities = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    result = estimator.compute(probabilities)

    np.testing.assert_allclose(
        result,
        [1.0, 1.0],
    )


def test_uniform_distribution(estimator):
    probabilities = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )

    result = estimator.compute(probabilities)

    np.testing.assert_allclose(
        result,
        [0.5, 0.5],
    )


def test_zero_probabilities_are_valid(estimator):
    probabilities = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    result = estimator.compute(probabilities)

    assert np.all(np.isfinite(result))


def test_batch_output_length(estimator):
    probabilities = np.array(
        [
            [0.7, 0.3],
            [0.2, 0.8],
            [0.4, 0.6],
        ]
    )

    result = estimator.compute(probabilities)

    assert result.shape == (3,)


def test_average(estimator):
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.6, 0.4],
        ]
    )

    assert estimator.average(probabilities) == pytest.approx(0.75)


def test_minimum(estimator):
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.6, 0.4],
        ]
    )

    assert estimator.minimum(probabilities) == pytest.approx(0.6)


def test_maximum(estimator):
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.6, 0.4],
        ]
    )

    assert estimator.maximum(probabilities) == pytest.approx(0.9)


def test_statistics(estimator):
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.6, 0.4],
        ]
    )

    result = estimator.statistics(probabilities)

    assert result["mean"] == pytest.approx(0.75)
    assert result["min"] == pytest.approx(0.6)
    assert result["max"] == pytest.approx(0.9)
    assert result["std"] == pytest.approx(0.15)


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([[np.nan, 1.0]]),
        np.array([[np.inf, 0.0]]),
        np.array([[-0.1, 1.1]]),
        np.array([[0.8, 0.8]]),
    ],
)
def test_invalid_probabilities(estimator, probabilities):
    with pytest.raises(ValidationError):
        estimator.compute(probabilities)


def test_empty_batch(estimator):
    probabilities = np.empty((0, 2))

    with pytest.raises(ValidationError):
        estimator.compute(probabilities)


def test_wrong_dimensions(estimator):
    probabilities = np.array([0.5, 0.5])

    with pytest.raises(ValidationError):
        estimator.compute(probabilities)