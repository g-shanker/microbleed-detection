"""Measure what a detection strategy and dilation do to a subject.

Dilating the vessel mask removes more of each vessel (and its blurry rim)
before inpainting, but it also risks erasing real microbleeds that touch or sit
near a vessel. Each strategy and iteration count is therefore judged on both
sides.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.ndimage import binary_dilation, label

from experiments.frst.scoring import SubjectScores, subject_scores
from microbleednet.core.transforms import frst, inpaint_vessels

from .detectors import DETECTORS
from .prepare import PreparedSubject


@dataclass(frozen=True)
class Measurement:
    """Raw counts for one subject, strategy and iteration count."""

    lesion_voxels: int
    lesion_voxels_removed: int
    lesions: int
    lesions_hit: int
    vessel_fraction: float
    residual_vessels: float
    scores: SubjectScores


@dataclass(frozen=True)
class Outcome:
    """A measurement with the arrays it came from, for saving."""

    measurement: Measurement
    mask: np.ndarray
    inpainted: np.ndarray


def dilate_in_plane(vessels: np.ndarray, iterations: int) -> np.ndarray:
    """Dilate every slice on its own, exactly like the pipeline does.

    ``iterations = 0`` leaves the mask unchanged (scipy would otherwise dilate
    until nothing changes).
    """
    if iterations == 0:
        return vessels.copy()
    dilated = np.zeros(vessels.shape, dtype=bool)
    for index in range(vessels.shape[2]):
        dilated[:, :, index] = binary_dilation(
            vessels[:, :, index], iterations=iterations
        )
    return dilated


def measure(
    prepared: PreparedSubject,
    strategy: str,
    iterations: int,
    *,
    lesion_margin: int,
    exclusion_radius: int,
    peak_min_distance: int,
    residual: bool = True,
) -> Outcome:
    """Inpaint with a strategy's mask and ``iterations`` of dilation, then measure.

    ``residual=False`` skips rerunning the detector on the inpainted volume, the
    slowest step, and reports ``residual_vessels`` as NaN.
    """
    vessels = prepared.vessels[strategy]
    mask = dilate_in_plane(vessels, iterations)
    inpainted = inpaint_vessels.inpaint_with_neighborhood_mean(prepared.volume, mask)

    components, count = cast(
        tuple[np.ndarray, int],
        label(prepared.lesions, structure=np.ones((3, 3, 3))),
    )
    hit = sum(
        bool((mask & (components == index)).any()) for index in range(1, count + 1)
    )
    brain_voxels = max(int(prepared.brain.sum()), 1)
    original_vessels = int((vessels & prepared.brain).sum())
    # Run the same detector again on what inpainting left.
    remaining_vessels = (
        int((DETECTORS[strategy](inpainted) & prepared.brain).sum()) if residual else 0
    )
    measurement = Measurement(
        lesion_voxels=int(prepared.lesions.sum()),
        lesion_voxels_removed=int((mask & prepared.lesions).sum()),
        lesions=count,
        lesions_hit=hit,
        vessel_fraction=float((mask & prepared.brain).sum() / brain_voxels),
        residual_vessels=(
            remaining_vessels / original_vessels
            if residual and original_vessels
            else float("nan")
        ),
        scores=subject_scores(
            frst.apply(inpainted),
            prepared.lesions,
            prepared.brain,
            lesion_margin=lesion_margin,
            exclusion_radius=exclusion_radius,
            peak_min_distance=peak_min_distance,
        ),
    )
    return Outcome(measurement, mask, inpainted)
