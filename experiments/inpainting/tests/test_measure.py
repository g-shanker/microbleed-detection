from typing import Any

import numpy as np
import pytest

from experiments.inpainting import detectors
from experiments.inpainting.measure import dilate_in_plane, measure
from experiments.inpainting.prepare import PreparedSubject
from experiments.inpainting.tests.support import (
    AFFINE,
    LESION,
    SHAPE,
    THROUGH_PLANE,
    VESSEL,
    base_volume,
)
from microbleednet.constants import VESSEL_DILATION_ITERATIONS
from microbleednet.core.transforms import inpaint_vessels

OPTIONS: dict[str, Any] = {
    "lesion_margin": 0,
    "exclusion_radius": 2,
    "peak_min_distance": 1,
}


@pytest.fixture
def threshold_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detect vessels as voxels brighter than 0.9, like the shared fake pipeline."""
    monkeypatch.setattr(
        inpaint_vessels, "get_slice_vessel_mask", lambda image: image > 0.9
    )


def voxel_mask(*indices: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=bool)
    for index in indices:
        mask[index] = True
    return mask


def prepared(
    *, vessels: np.ndarray, volume: np.ndarray | None = None, **other: np.ndarray
) -> PreparedSubject:
    return PreparedSubject(
        volume=base_volume() if volume is None else volume,
        lesions=voxel_mask(LESION),
        brain=np.ones(SHAPE, dtype=bool),
        affine=AFFINE,
        spacing=(1.0, 1.0, 3.0),
        vessels={"axial": vessels, **other},
    )


def run(subject: PreparedSubject, iterations: int, strategy: str = "axial"):
    return measure(subject, strategy, iterations, **OPTIONS)


def test_zero_iterations_leave_the_mask_unchanged() -> None:
    vessels = voxel_mask(VESSEL)

    dilated = dilate_in_plane(vessels, 0)

    np.testing.assert_array_equal(dilated, vessels)
    assert dilated is not vessels


def test_dilation_grows_within_each_slice_only() -> None:
    dilated = dilate_in_plane(voxel_mask(VESSEL), 2)

    assert dilated[VESSEL[0], VESSEL[1] + 2, VESSEL[2]]
    assert not dilated[VESSEL[0], VESSEL[1] + 3, VESSEL[2]]
    assert not dilated[:, :, VESSEL[2] - 1].any()
    assert not dilated[:, :, VESSEL[2] + 1].any()


def test_dilation_matches_the_pipelines_vessel_mask(
    monkeypatch: pytest.MonkeyPatch, threshold_detector: None
) -> None:
    class Serial:
        def __init__(self, **options) -> None:
            pass

        def __call__(self, tasks):
            return (function(*args, **kwargs) for function, args, kwargs in tasks)

    monkeypatch.setattr(inpaint_vessels, "Parallel", Serial)
    volume = base_volume()

    pipeline = inpaint_vessels.get_volume_vessel_mask(volume)

    np.testing.assert_array_equal(
        dilate_in_plane(volume > 0.9, VESSEL_DILATION_ITERATIONS), pipeline > 0
    )


def test_a_lesion_beyond_the_dilation_reach_is_kept(threshold_detector: None) -> None:
    result = run(prepared(vessels=voxel_mask(VESSEL)), 1).measurement

    assert (result.lesions, result.lesions_hit) == (1, 0)
    assert (result.lesion_voxels, result.lesion_voxels_removed) == (1, 0)


def test_a_lesion_within_the_dilation_reach_is_erased(threshold_detector: None) -> None:
    result = run(prepared(vessels=voxel_mask(VESSEL)), 2).measurement

    assert (result.lesions, result.lesions_hit) == (1, 1)
    assert (result.lesion_voxels, result.lesion_voxels_removed) == (1, 1)


def test_vessel_fraction_is_the_dilated_share_of_the_brain(
    threshold_detector: None,
) -> None:
    result = run(prepared(vessels=voxel_mask(VESSEL)), 0).measurement

    assert result.vessel_fraction == pytest.approx(1 / np.prod(SHAPE))


def test_inpainted_vessels_are_not_detected_again(threshold_detector: None) -> None:
    result = run(prepared(vessels=voxel_mask(VESSEL)), 0).measurement

    assert result.residual_vessels == 0.0


def test_vessels_the_mask_missed_are_detected_again(threshold_detector: None) -> None:
    other = (3, 3, 1)
    volume = base_volume()
    volume[other] = 1.0

    result = run(prepared(vessels=voxel_mask(VESSEL), volume=volume), 0).measurement

    assert result.residual_vessels == 1.0


def test_residual_vessels_are_undefined_without_original_vessels(
    threshold_detector: None,
) -> None:
    result = run(prepared(vessels=voxel_mask()), 0).measurement

    assert np.isnan(result.residual_vessels)


def test_scores_cover_the_lesion_and_the_background(threshold_detector: None) -> None:
    result = run(prepared(vessels=voxel_mask(VESSEL)), 0).measurement

    assert len(result.scores.lesion_scores) == 1
    assert len(result.scores.background_peaks) > 0


def test_the_outcome_carries_the_dilated_mask_and_the_inpainted_volume(
    threshold_detector: None,
) -> None:
    outcome = run(prepared(vessels=voxel_mask(VESSEL)), 1)

    np.testing.assert_array_equal(outcome.mask, dilate_in_plane(voxel_mask(VESSEL), 1))
    assert outcome.inpainted.shape == SHAPE
    assert not np.array_equal(outcome.inpainted, base_volume())


def test_each_strategy_is_measured_with_its_own_vessel_mask(
    threshold_detector: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        detectors.DETECTORS, "frangi3d", lambda volume: np.zeros_like(volume, bool)
    )
    subject = prepared(
        vessels=voxel_mask(VESSEL), frangi3d=voxel_mask(VESSEL, THROUGH_PLANE)
    )

    axial = run(subject, 0, "axial")
    three_d = run(subject, 0, "frangi3d")

    assert axial.mask.sum() == 1
    assert three_d.mask.sum() == 2


def test_residual_vessels_use_the_strategys_own_detector(
    threshold_detector: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []
    monkeypatch.setitem(
        detectors.DETECTORS,
        "frangi3d",
        lambda volume: seen.append(volume) or np.zeros_like(volume, bool),
    )
    subject = prepared(vessels=voxel_mask(VESSEL), frangi3d=voxel_mask(VESSEL))

    result = run(subject, 0, "frangi3d").measurement

    assert len(seen) == 1
    assert result.residual_vessels == 0.0


def test_skipping_the_residual_check_does_not_run_the_detector(
    threshold_detector: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        detectors.DETECTORS, "axial", lambda _: pytest.fail("detector ran")
    )
    subject = prepared(vessels=voxel_mask(VESSEL))

    result = measure(subject, "axial", 1, residual=False, **OPTIONS).measurement

    assert np.isnan(result.residual_vessels)
    assert result.vessel_fraction > 0 and len(result.scores.lesion_scores) == 1
