import json

import pytest

from aegis.core.enums import Recommendation, RiskLevel
from aegis.core.report import PredictionReport


def build_report() -> PredictionReport:
    return PredictionReport(
        prediction=1,
        confidence=0.91,
        uncertainty=0.12,
        ood=False,
        ood_score=0.18,
        trust_score=0.86,
        risk_level=RiskLevel.LOW,
        recommendation=Recommendation.AUTO_APPROVE,
        metadata={
            "ood_raw_score": 0.54,
            "ood_threshold": 3.0,
        },
    )


def test_report_creation():
    report = build_report()

    assert report.prediction == 1
    assert report.confidence == pytest.approx(0.91)
    assert report.uncertainty == pytest.approx(0.12)
    assert report.ood is False
    assert report.ood_score == pytest.approx(0.18)
    assert report.trust_score == pytest.approx(0.86)
    assert report.risk_level == RiskLevel.LOW
    assert report.recommendation == Recommendation.AUTO_APPROVE


def test_report_to_dict():
    report = build_report()

    result = report.to_dict()

    assert isinstance(result, dict)
    assert result["prediction"] == 1
    assert result["confidence"] == pytest.approx(0.91)
    assert result["trust_score"] == pytest.approx(0.86)


def test_report_to_json():
    report = build_report()

    result = report.to_json()

    parsed = json.loads(result)

    assert parsed["prediction"] == 1
    assert parsed["ood"] is False
    assert parsed["risk_level"] == "LOW"


def test_report_summary():
    report = build_report()

    summary = report.summary()

    assert "Prediction=1" in summary
    assert "Confidence=0.910" in summary
    assert "Uncertainty=0.120" in summary
    assert "OOD=False" in summary
    assert "Trust=0.860" in summary
    assert "Risk=LOW" in summary
    assert "Auto Approve" in summary


def test_report_string_is_json():
    report = build_report()

    result = str(report)

    parsed = json.loads(result)

    assert parsed["prediction"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("uncertainty", -0.1),
        ("uncertainty", 1.1),
        ("ood_score", -0.1),
        ("ood_score", 1.1),
        ("trust_score", -0.1),
        ("trust_score", 1.1),
    ],
)
def test_report_rejects_invalid_scores(field, value):
    values = {
        "prediction": 1,
        "confidence": 0.9,
        "uncertainty": 0.1,
        "ood": False,
        "ood_score": 0.1,
        "trust_score": 0.9,
        "risk_level": RiskLevel.LOW,
        "recommendation": Recommendation.AUTO_APPROVE,
    }

    values[field] = value

    with pytest.raises(ValueError):
        PredictionReport(**values)


def test_report_rejects_unknown_fields():
    values = {
        "prediction": 1,
        "confidence": 0.9,
        "uncertainty": 0.1,
        "ood": False,
        "ood_score": 0.1,
        "trust_score": 0.9,
        "risk_level": RiskLevel.LOW,
        "recommendation": Recommendation.AUTO_APPROVE,
        "fake_metric": 123,
    }

    with pytest.raises(ValueError):
        PredictionReport(**values)


def test_report_allows_metadata():
    report = build_report()

    assert report.metadata["ood_raw_score"] == pytest.approx(0.54)
    assert report.metadata["ood_threshold"] == pytest.approx(3.0)


def test_negative_json_indent_is_rejected():
    report = build_report()

    with pytest.raises(ValueError):
        report.to_json(indent=-1)