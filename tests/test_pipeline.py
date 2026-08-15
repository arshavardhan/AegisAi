import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from aegis import AegisModel
from aegis.core.enums import Recommendation, RiskLevel


def build_model():
    X, y = make_classification(
        n_samples=160,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )
    classifier = LogisticRegression(max_iter=500, random_state=42)
    classifier.fit(X, y)
    return classifier, X


def test_end_to_end_prediction_report():
    classifier, X = build_model()
    model = AegisModel(classifier).fit(X)

    report = model.predict(X[:1])

    assert 0.0 <= report.confidence <= 1.0
    assert 0.0 <= report.uncertainty <= 1.0
    assert 0.0 <= report.ood_score <= 1.0
    assert 0.0 <= report.trust_score <= 1.0
    assert report.risk_level in set(RiskLevel)
    assert report.recommendation in set(Recommendation)


def test_batch_prediction_returns_one_report_per_sample():
    classifier, X = build_model()
    model = AegisModel(classifier).fit(X)

    reports = model.predict_batch(X[:5])

    assert len(reports) == 5
    assert all(0.0 <= report.trust_score <= 1.0 for report in reports)


def test_ood_feature_mismatch_is_reported():
    classifier, X = build_model()
    model = AegisModel(classifier).fit(X)

    try:
        model.predict(np.zeros((1, X.shape[1] + 1)))
    except Exception as exc:
        assert "features" in str(exc).lower()
    else:
        raise AssertionError("Expected feature mismatch to fail")
