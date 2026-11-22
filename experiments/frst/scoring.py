"""Score how well a FRST map alone separates annotated microbleeds from clutter.

For each subject the FRST map yields one score per annotated lesion (its peak
response) and a set of background peaks (local maxima away from every lesion).
Scores are only compared within a subject, so the metrics do not depend on the
absolute scale of a variant's FRST values.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, find_objects, label
from skimage.feature import peak_local_max


@dataclass(frozen=True)
class SubjectScores:
    """Lesion scores and background peak values for one subject."""

    lesion_scores: np.ndarray
    background_peaks: np.ndarray


def subject_scores(
    frst_map: np.ndarray,
    lesion_mask: np.ndarray,
    brain_mask: np.ndarray,
    *,
    lesion_margin: int,
    exclusion_radius: int,
    peak_min_distance: int,
) -> SubjectScores:
    """Score every lesion and collect the background peaks of one subject.

    ``lesion_margin`` widens each lesion before taking its peak, which absorbs
    small misalignments. Background peaks must lie in the brain and more than
    ``exclusion_radius`` voxels from any lesion, at least ``peak_min_distance``
    voxels apart.
    """
    values = np.asarray(frst_map, dtype=np.float64)
    lesions = np.asarray(lesion_mask) > 0
    labeled, _ = cast(
        tuple[np.ndarray, int], label(lesions, structure=np.ones((3, 3, 3)))
    )
    boxes = cast(list[tuple[slice, ...] | None], find_objects(labeled))
    lesion_scores = []
    for index, box in enumerate(boxes, start=1):
        if box is None:
            continue
        window = tuple(
            slice(max(axis.start - lesion_margin, 0), axis.stop + lesion_margin)
            for axis in box
        )
        region = np.asarray(labeled[window] == index, dtype=bool)
        if lesion_margin > 0:
            region = np.asarray(
                binary_dilation(region, iterations=lesion_margin), dtype=bool
            )
        peak = np.max(values[window], where=region, initial=-np.inf)
        lesion_scores.append(float(peak))

    eligible = np.asarray(brain_mask) > 0
    if lesions.any():
        eligible &= np.asarray(distance_transform_edt(~lesions)) > exclusion_radius
    peaks = np.asarray(
        peak_local_max(
            values,
            min_distance=peak_min_distance,
            threshold_abs=float(np.finfo(np.float32).tiny),
            labels=eligible.astype(np.uint8),
            exclude_border=False,
        ),
        dtype=np.intp,
    )
    return SubjectScores(
        np.asarray(lesion_scores, dtype=np.float64),
        values[peaks[:, 0], peaks[:, 1], peaks[:, 2]],
    )


def summarize(
    subjects: list[SubjectScores], recall_at: tuple[int, ...]
) -> dict[str, float]:
    """Pool lesions across subjects into the metrics used to rank variants.

    * ``mean_auc``: for each lesion, the share of its subject's background peaks
      that score lower (ties count half), averaged over lesions.
    * ``recall@K``: share of lesions that fewer than K background peaks of their
      subject match or exceed.
    """
    aucs: list[float] = []
    found = dict.fromkeys(recall_at, 0)
    for subject in subjects:
        peaks = np.sort(subject.background_peaks)
        for score in subject.lesion_scores:
            below = int(np.searchsorted(peaks, score, side="left"))
            tied = int(np.searchsorted(peaks, score, side="right")) - below
            aucs.append((below + tied / 2) / len(peaks) if len(peaks) else 1.0)
            at_or_above = len(peaks) - below
            for count in recall_at:
                found[count] += at_or_above < count

    lesions = len(aucs)
    summary = {"lesions": float(lesions)}
    summary["mean_auc"] = float(np.mean(aucs)) if lesions else float("nan")
    for count in recall_at:
        summary[f"recall@{count}"] = found[count] / lesions if lesions else float("nan")
    return summary
