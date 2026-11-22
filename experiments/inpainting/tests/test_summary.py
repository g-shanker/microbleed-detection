import numpy as np
import pytest

from experiments.frst.scoring import SubjectScores
from experiments.inpainting.measure import Measurement
from experiments.inpainting.summary import recommend, summarize_iterations


def measurement(**overrides) -> Measurement:
    values = {
        "lesion_voxels": 10,
        "lesion_voxels_removed": 0,
        "lesions": 2,
        "lesions_hit": 0,
        "vessel_fraction": 0.1,
        "residual_vessels": 0.5,
        "scores": SubjectScores(np.array([0.9]), np.array([0.5, 1.0])),
    }
    values.update(overrides)
    return Measurement(**values)


def row(iterations: int, lesion_loss: float, mean_auc: float) -> dict[str, float]:
    return {
        "iterations": iterations,
        "lesion_loss": lesion_loss,
        "mean_auc": mean_auc,
    }


def test_lesion_loss_pools_voxels_across_subjects() -> None:
    result = summarize_iterations(
        [
            measurement(lesion_voxels=10, lesion_voxels_removed=1),
            measurement(lesion_voxels=30, lesion_voxels_removed=3),
        ],
        recall_at=(1,),
    )

    assert result["lesion_loss"] == pytest.approx(4 / 40)


def test_lesions_hit_pools_lesions_across_subjects() -> None:
    result = summarize_iterations(
        [measurement(lesions=2, lesions_hit=1), measurement(lesions=2, lesions_hit=0)],
        recall_at=(1,),
    )

    assert result["lesions_hit"] == 0.25


def test_vessel_metrics_average_over_subjects() -> None:
    result = summarize_iterations(
        [
            measurement(vessel_fraction=0.1, residual_vessels=0.2),
            measurement(vessel_fraction=0.3, residual_vessels=0.4),
        ],
        recall_at=(1,),
    )

    assert result["vessel_fraction"] == pytest.approx(0.2)
    assert result["residual_vessels"] == pytest.approx(0.3)


def test_undefined_residuals_are_ignored_in_the_average() -> None:
    result = summarize_iterations(
        [measurement(residual_vessels=float("nan")), measurement(residual_vessels=0.4)],
        recall_at=(1,),
    )

    assert result["residual_vessels"] == pytest.approx(0.4)


def test_separation_metrics_come_from_the_frst_scores() -> None:
    result = summarize_iterations([measurement()], recall_at=(1, 2))

    assert result["mean_auc"] == 0.5
    assert (result["recall@1"], result["recall@2"]) == (0.0, 1.0)


def test_no_lesions_gives_undefined_loss() -> None:
    result = summarize_iterations(
        [
            measurement(
                lesion_voxels=0,
                lesions=0,
                scores=SubjectScores(np.array([]), np.array([1.0])),
            )
        ],
        recall_at=(1,),
    )

    assert np.isnan(result["lesion_loss"]) and np.isnan(result["lesions_hit"])


def test_recommendation_is_the_smallest_count_near_the_best_auc() -> None:
    rows = [
        row(0, 0.0, 0.70),
        row(1, 0.0, 0.80),
        row(2, 0.01, 0.805),
        row(3, 0.02, 0.81),
    ]

    assert recommend(rows, max_lesion_loss=0.05, auc_tolerance=0.01) == 1


def test_counts_that_erase_too_many_lesions_are_not_recommended() -> None:
    rows = [row(0, 0.0, 0.70), row(1, 0.0, 0.75), row(2, 0.2, 0.95)]

    assert recommend(rows, max_lesion_loss=0.05, auc_tolerance=0.01) == 1


def test_no_recommendation_when_every_count_erases_too_much() -> None:
    assert (
        recommend([row(1, 0.5, 0.9)], max_lesion_loss=0.05, auc_tolerance=0.01) is None
    )


def test_undefined_auc_is_never_recommended() -> None:
    rows = [row(0, 0.0, float("nan")), row(1, 0.0, 0.6)]

    assert recommend(rows, max_lesion_loss=0.05, auc_tolerance=0.0) == 1


def test_residuals_that_are_all_undefined_average_to_nan_without_a_warning(
    recwarn,
) -> None:
    result = summarize_iterations(
        [measurement(residual_vessels=float("nan"))] * 2, recall_at=(1,)
    )

    assert np.isnan(result["residual_vessels"])
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
