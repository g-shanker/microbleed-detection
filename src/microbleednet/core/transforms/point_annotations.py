"""Dilation of point-annotated lesion masks into physical-radius balls.

Self-contained so it can be removed with its call site in
``orchestration/pipes/preprocess.py`` and the ``point_annotation_radius_mm``
config and manifest fields.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

from ...constants import COMPONENT_CONNECTIVITY
from ..datamodels import VoxelSpacing
from ..utils import label_components
from .patch import get_target_centers


def dilate_point_annotations(
    mask: np.ndarray, voxel_sizes: VoxelSpacing, radius_mm: float
) -> np.ndarray:
    """Replace each annotated point with a ball of the given physical radius."""
    centers = np.zeros(mask.shape, dtype=bool)
    for center in get_target_centers(
        label_components(mask > 0, COMPONENT_CONNECTIVITY)
    ):
        centers[center] = True
    if not centers.any():
        return np.zeros(mask.shape, dtype=np.uint8)
    distance_mm = distance_transform_edt(~centers, sampling=voxel_sizes)
    return (np.asarray(distance_mm) <= radius_mm).astype(np.uint8)
