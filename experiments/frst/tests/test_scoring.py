import numpy as np
import pytest

from experiments.frst.scoring import SubjectScores, subject_scores, summarize

SHAPE = (20, 20, 3)
OPTIONS = {"lesion_margin": 0, "exclusion_radius": 2, "peak_min_distance": 1}


def frst_map(**peaks: float) -> np.ndarray:
    """A map with one peak per ``x_y`` keyword, all in the middle slice."""
    values = np.zeros(SHAPE, dtype=np.float32)
    for position, value in peaks.items():
        x, y = (int(part) for part in position.removeprefix("p").split("_"))
        values[x, y, 1] = value
    return values


def lesion_at(x: int, y: int) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[x, y, 1] = 1
    return mask


def brain() -> np.ndarray:
    return np.ones(SHAPE, dtype=bool)


def test_scores_the_lesion_peak_and_every_distant_background_peak() -> None:
    values = frst_map(p5_5=0.9, p15_15=0.5, p15_5=1.0)

    scores = subject_scores(values, lesion_at(5, 5), brain(), **OPTIONS)

    np.testing.assert_allclose(scores.lesion_scores, [0.9])
    np.testing.assert_allclose(np.sort(scores.background_peaks), [0.5, 1.0])


def test_peaks_next_to_a_lesion_are_not_background() -> None:
    values = frst_map(p5_5=0.9, p6_5=5.0, p15_15=0.5)

    scores = subject_scores(values, lesion_at(5, 5), brain(), **OPTIONS)

    np.testing.assert_allclose(scores.background_peaks, [0.5])


def test_lesion_margin_collects_the_peak_beside_an_annotation() -> None:
    values = frst_map(p6_5=0.8)

    narrow = subject_scores(values, lesion_at(5, 5), brain(), **OPTIONS)
    wide = subject_scores(
        values, lesion_at(5, 5), brain(), **{**OPTIONS, "lesion_margin": 1}
    )

    assert narrow.lesion_scores[0] == 0
    assert wide.lesion_scores[0] == pytest.approx(0.8)


def test_peaks_outside_the_brain_are_ignored() -> None:
    values = frst_map(p15_15=0.5, p2_2=0.7)
    mask = brain()
    mask[:5, :5] = False

    scores = subject_scores(values, lesion_at(10, 10), mask, **OPTIONS)

    np.testing.assert_allclose(scores.background_peaks, [0.5])


def test_every_lesion_component_gets_its_own_score() -> None:
    values = frst_map(p5_5=0.9, p15_15=0.4)
    lesions = lesion_at(5, 5) + lesion_at(15, 15)

    scores = subject_scores(values, lesions, brain(), **OPTIONS)

    np.testing.assert_allclose(np.sort(scores.lesion_scores), [0.4, 0.9])


def subject(lesions: list[float], peaks: list[float]) -> SubjectScores:
    return SubjectScores(np.array(lesions), np.array(peaks))


def test_auc_is_the_share_of_background_peaks_below_the_lesion() -> None:
    result = summarize([subject([0.9], [0.5, 1.0])], recall_at=(1,))

    assert result["mean_auc"] == 0.5


def test_tied_peaks_count_half_towards_auc() -> None:
    result = summarize([subject([0.5], [0.5, 0.1])], recall_at=(1,))

    assert result["mean_auc"] == 0.75


def test_recall_at_k_needs_fewer_than_k_peaks_at_or_above_the_lesion() -> None:
    result = summarize([subject([0.9], [0.5, 1.0])], recall_at=(1, 2))

    assert result["recall@1"] == 0.0
    assert result["recall@2"] == 1.0


def test_metrics_pool_lesions_across_subjects() -> None:
    result = summarize([subject([2.0], [1.0]), subject([0.0], [1.0])], recall_at=(1,))

    assert result["lesions"] == 2
    assert result["mean_auc"] == 0.5
    assert result["recall@1"] == 0.5


def test_a_subject_without_background_peaks_scores_perfectly() -> None:
    result = summarize([subject([0.1], [])], recall_at=(1,))

    assert result["mean_auc"] == 1.0
    assert result["recall@1"] == 1.0


def test_no_lesions_yields_nan_metrics() -> None:
    result = summarize([subject([], [1.0])], recall_at=(1,))

    assert result["lesions"] == 0
    assert np.isnan(result["mean_auc"])
    assert np.isnan(result["recall@1"])
