import numpy as np
import pytest

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from aegis.core.exceptions import (
    ModelCompatibilityError,
    PredictionError,
)
from aegis.wrappers.sklearn_wrapper import SklearnWrapper


def build_classifier():
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )

    model = LogisticRegression(
        max_iter=500,
        random_state=42,
    )

    model.fit(X, y)

    return model, X


def test_sklearn_wrapper_framework():
    model, _ = build_classifier()

    wrapper = SklearnWrapper(model)

    assert wrapper.framework == "scikit-learn"


def test_sklearn_wrapper_validates_fitted_classifier():
    model, _ = build_classifier()

    wrapper = SklearnWrapper(model)

    wrapper.validate()


def test_sklearn_wrapper_predict():
    model, X = build_classifier()

    wrapper = SklearnWrapper(model)

    predictions = wrapper.predict(X[:5])

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (5,)


def test_sklearn_wrapper_predict_proba():
    model, X = build_classifier()

    wrapper = SklearnWrapper(model)

    probabilities = wrapper.predict_proba(X[:5])

    assert probabilities.shape == (5, 2)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )


def test_sklearn_wrapper_confidence():
    model, X = build_classifier()

    wrapper = SklearnWrapper(model)

    confidence = wrapper.predict_confidence(
        X[:5]
    )

    probabilities = wrapper.predict_proba(
        X[:5]
    )

    np.testing.assert_allclose(
        confidence,
        probabilities.max(axis=1),
    )


def test_sklearn_wrapper_classes():
    model, _ = build_classifier()

    wrapper = SklearnWrapper(model)

    classes = wrapper.classes()

    assert classes.shape == (2,)


def test_unfitted_classifier_is_rejected():
    model = LogisticRegression(
        max_iter=500,
    )

    with pytest.raises(
        ModelCompatibilityError,
        match="fitted",
    ):
        SklearnWrapper(model)


def test_model_without_predict_proba_is_rejected():
    model = SVC(
        probability=False,
    )

    X, y = make_classification(
        n_samples=50,
        n_features=4,
        random_state=42,
    )

    model.fit(X, y)

    with pytest.raises(
        ModelCompatibilityError,
        match="predict_proba",
    ):
        SklearnWrapper(model)


def test_none_model_is_rejected():
    with pytest.raises(
        ModelCompatibilityError,
        match="cannot be None",
    ):
        SklearnWrapper(None)


def test_probability_output_is_validated():
    model, _ = build_classifier()

    wrapper = SklearnWrapper(model)

    valid = np.array(
        [
            [0.2, 0.8],
            [0.7, 0.3],
        ]
    )

    wrapper._validate_probabilities(valid)


def test_invalid_probability_range_is_rejected():
    model, _ = build_classifier()

    wrapper = SklearnWrapper(model)

    invalid = np.array(
        [
            [1.2, -0.2],
        ]
    )

    with pytest.raises(
        PredictionError,
        match="within",
    ):
        wrapper._validate_probabilities(invalid)


def test_probability_rows_must_sum_to_one():
    model, _ = build_classifier()

    wrapper = SklearnWrapper(model)

    invalid = np.array(
        [
            [0.2, 0.2],
        ]
    )

    with pytest.raises(
        PredictionError,
        match="sum to 1",
    ):
        wrapper._validate_probabilities(invalid)


def test_probability_nan_is_rejected():
    model, _ = build_classifier()

    wrapper = SklearnWrapper(model)

    invalid = np.array(
        [
            [np.nan, np.nan],
        ]
    )

    with pytest.raises(
        PredictionError,
        match="NaN",
    ):
        wrapper._validate_probabilities(invalid)