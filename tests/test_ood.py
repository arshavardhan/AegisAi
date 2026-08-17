import numpy as np
import pytest

from aegis.core.exceptions import OODDetectionError
from aegis.ood.zscore import ZScoreOODDetector


def reference_data():
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [1.5, 0.5],
            [0.5, 1.5],
        ]
    )


def test_detector_name():
    detector = ZScoreOODDetector()

    assert detector.name == "Z-Score OOD Detector"


def test_detector_starts_unfitted():
    detector = ZScoreOODDetector()

    assert detector.is_fitted is False


def test_detector_becomes_fitted():
    detector = ZScoreOODDetector()

    detector.fit(reference_data())

    assert detector.is_fitted is True


def test_fit_returns_detector():
    detector = ZScoreOODDetector()

    result = detector.fit(reference_data())

    assert result is detector


def test_in_distribution_samples_have_low_scores():
    detector = ZScoreOODDetector(
        threshold=3.0
    ).fit(reference_data())

    scores = detector.score(
        np.array(
            [
                [1.0, 1.0],
                [1.1, 0.9],
            ]
        )
    )

    assert scores.shape == (2,)
    assert np.all(scores >= 0.0)
    assert np.all(scores < 3.0)


def test_extreme_outlier_has_high_score():
    detector = ZScoreOODDetector(
        threshold=3.0
    ).fit(reference_data())

    scores = detector.score(
        np.array(
            [
                [100.0, 100.0],
            ]
        )
    )

    assert scores.shape == (1,)
    assert scores[0] > 3.0


def test_predict_returns_boolean_array():
    detector = ZScoreOODDetector(
        threshold=3.0
    ).fit(reference_data())

    result = detector.predict(
        np.array(
            [
                [1.0, 1.0],
                [100.0, 100.0],
            ]
        )
    )

    assert result.dtype == bool
    assert result.tolist() == [False, True]


def test_score_is_raw_unbounded_distance():
    detector = ZScoreOODDetector(
        threshold=3.0
    ).fit(reference_data())

    score = detector.score(
        np.array(
            [
                [1000.0, 1000.0],
            ]
        )
    )[0]

    assert score > 1.0
    assert score > detector.threshold


def test_batch_score_returns_one_score_per_sample():
    detector = ZScoreOODDetector().fit(
        reference_data()
    )

    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [100.0, 100.0],
        ]
    )

    scores = detector.score(X)

    assert scores.shape == (4,)


def test_batch_predict_returns_one_flag_per_sample():
    detector = ZScoreOODDetector().fit(
        reference_data()
    )

    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [100.0, 100.0],
        ]
    )

    result = detector.predict(X)

    assert result.shape == (3,)
    assert result.dtype == bool


def test_single_feature_vector_is_supported():
    detector = ZScoreOODDetector().fit(
        np.array(
            [
                [0.0, 1.0],
                [1.0, 2.0],
                [2.0, 3.0],
            ]
        )
    )

    result = detector.score(
        np.array([1.0, 2.0])
    )

    assert result.shape == (1,)


def test_unfitted_score_fails():
    detector = ZScoreOODDetector()

    with pytest.raises(
        OODDetectionError,
        match="not been fitted",
    ):
        detector.score(
            np.array([[1.0, 1.0]])
        )


def test_unfitted_predict_fails():
    detector = ZScoreOODDetector()

    with pytest.raises(
        OODDetectionError,
        match="not been fitted",
    ):
        detector.predict(
            np.array([[1.0, 1.0]])
        )


def test_feature_count_mismatch_fails():
    detector = ZScoreOODDetector().fit(
        reference_data()
    )

    with pytest.raises(
        OODDetectionError,
        match="Feature count mismatch",
    ):
        detector.score(
            np.zeros((1, 3))
        )


def test_empty_reference_data_fails():
    detector = ZScoreOODDetector()

    with pytest.raises(
        OODDetectionError,
        match="zero samples",
    ):
        detector.fit(
            np.empty((0, 2))
        )


def test_single_reference_sample_fails():
    detector = ZScoreOODDetector()

    with pytest.raises(
        OODDetectionError,
        match="At least two reference samples",
    ):
        detector.fit(
            np.array([[1.0, 1.0]])
        )


@pytest.mark.parametrize(
    "bad_input",
    [
        np.array([[np.nan, 1.0]]),
        np.array([[np.inf, 1.0]]),
        np.array([[-np.inf, 1.0]]),
    ],
)
def test_invalid_input_fails(bad_input):
    detector = ZScoreOODDetector().fit(
        reference_data()
    )

    with pytest.raises(
        OODDetectionError,
        match="NaN or infinite",
    ):
        detector.score(bad_input)


@pytest.mark.parametrize(
    "threshold",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_threshold_fails(threshold):
    with pytest.raises(
        OODDetectionError,
        match="threshold",
    ):
        ZScoreOODDetector(
            threshold=threshold
        )


def test_statistics_returns_fitted_parameters():
    detector = ZScoreOODDetector(
        threshold=2.5
    ).fit(reference_data())

    statistics = detector.statistics()

    assert "mean" in statistics
    assert "std" in statistics
    assert "threshold" in statistics

    assert statistics["threshold"] == pytest.approx(2.5)
    assert statistics["mean"].shape == (2,)
    assert statistics["std"].shape == (2,)


def test_statistics_returns_copies():
    detector = ZScoreOODDetector().fit(
        reference_data()
    )

    statistics = detector.statistics()

    original_mean = detector.statistics()["mean"].copy()

    statistics["mean"][0] = 9999.0

    np.testing.assert_array_equal(
        detector.statistics()["mean"],
        original_mean,
    )


def test_statistics_before_fit_fails():
    detector = ZScoreOODDetector()

    with pytest.raises(
        OODDetectionError,
        match="not been fitted",
    ):
        detector.statistics()


def test_fit_predict():
    detector = ZScoreOODDetector(
        threshold=3.0
    )

    result = detector.fit_predict(
        np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ]
        )
    )

    assert detector.is_fitted is True
    assert result.shape == (3,)
    assert result.dtype == bool


def test_constant_features_do_not_create_infinite_scores():
    X = np.array(
        [
            [1.0, 10.0],
            [1.0, 20.0],
            [1.0, 30.0],
        ]
    )

    detector = ZScoreOODDetector().fit(X)

    scores = detector.score(
        np.array(
            [
                [1.0, 20.0],
                [2.0, 20.0],
            ]
        )
    )

    assert np.all(np.isfinite(scores))