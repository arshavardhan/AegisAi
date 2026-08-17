import numpy as np
import pytest

from aegis.core.exceptions import ValidationError
from aegis.uncertainty.entropy import EntropyEstimator


@pytest.fixture
def estimator():
    return EntropyEstimator()


def test_name(estimator):
    assert estimator.name == "Normalized Shannon Entropy"


def test_perfectly_certain_distribution(estimator):
    probabilities = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    result = estimator.compute(probabilities)

    np.testing.assert_allclose(
        result,
        [0.0, 0.0],
    )


def test_binary_uniform_distribution(estimator):
    probabilities = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )

    result = estimator.compute(probabilities)

    np.testing.assert_allclose(
        result,
        [1.0, 1.0],
    )


def test_binary_partial_distribution(estimator):
    probabilities = np.array(
        [
            [0.9, 0.1],
        ]
    )

    result = estimator.compute(probabilities)

    expected = -(
        0.9 * np.log2(0.9)
        + 0.1 * np.log2(0.1)
    )

    np.testing.assert_allclose(
        result,
        [expected],
    )


def test_multiclass_uniform_distribution(estimator):
    probabilities = np.array(
        [
            [1 / 3, 1 / 3, 1 / 3],
        ]
    )

    result = estimator.compute(probabilities)

    np.testing.assert_allclose(
        result,
        [1.0],
    )


def test_zero_probabilities_are_handled_correctly(estimator):
    probabilities = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    result = estimator.compute(probabilities)

    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(
        result,
        [0.0, 0.0],
    )


def test_entropy_is_between_zero_and_one(estimator):
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.7, 0.3],
            [0.5, 0.5],
        ]
    )

    result = estimator.compute(probabilities)

    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


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
            [1.0, 0.0],
            [0.5, 0.5],
        ]
    )

    assert estimator.average(probabilities) == pytest.approx(0.5)


def test_minimum(estimator):
    probabilities = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
        ]
    )

    assert estimator.minimum(probabilities) == pytest.approx(0.0)


def test_maximum(estimator):
    probabilities = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
        ]
    )

    assert estimator.maximum(probabilities) == pytest.approx(1.0)


def test_statistics(estimator):
    probabilities = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
        ]
    )

    result = estimator.statistics(probabilities)

    assert result["mean"] == pytest.approx(0.5)
    assert result["min"] == pytest.approx(0.0)
    assert result["max"] == pytest.approx(1.0)


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