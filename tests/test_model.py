import numpy as np
import pytest

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from aegis import AegisModel
from aegis.core.exceptions import (
    ModelCompatibilityError,
    PredictionError,
)


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


def test_aegis_model_wraps_sklearn_classifier():
    classifier, _ = build_classifier()

    model = AegisModel(classifier)

    assert model.model is classifier
    assert model.wrapper.framework == "scikit-learn"
    assert model.is_fitted is False


def test_aegis_fit_fits_reliability_components():
    classifier, X = build_classifier()

    model = AegisModel(classifier)

    result = model.fit(X)

    assert result is model
    assert model.is_fitted is True


def test_aegis_predict_requires_fit():
    classifier, X = build_classifier()

    model = AegisModel(classifier)

    with pytest.raises(
        PredictionError,
        match="not been fitted",
    ):
        model.predict(X[:1])


def test_aegis_predict_returns_report():
    classifier, X = build_classifier()

    model = AegisModel(classifier).fit(X)

    report = model.predict(X[:1])

    assert report.prediction in classifier.classes_
    assert 0.0 <= report.confidence <= 1.0
    assert 0.0 <= report.uncertainty <= 1.0
    assert 0.0 <= report.ood_score <= 1.0
    assert 0.0 <= report.trust_score <= 1.0


def test_aegis_predict_batch():
    classifier, X = build_classifier()

    model = AegisModel(classifier).fit(X)

    reports = model.predict_batch(X[:10])

    assert len(reports) == 10


def test_none_model_is_rejected():
    with pytest.raises(
        ModelCompatibilityError,
        match="cannot be None",
    ):
        AegisModel(None)


def test_unsupported_model_is_rejected():
    class FakeModel:
        pass

    with pytest.raises(
        ModelCompatibilityError,
        match="Unsupported model",
    ):
        AegisModel(FakeModel())


def test_pipeline_property_is_accessible():
    classifier, _ = build_classifier()

    model = AegisModel(classifier)

    assert model.pipeline is not None


def test_repr_contains_model_information():
    classifier, _ = build_classifier()

    model = AegisModel(classifier)

    representation = repr(model)

    assert "AegisModel" in representation
    assert "LogisticRegression" in representation
    assert "scikit-learn" in representation