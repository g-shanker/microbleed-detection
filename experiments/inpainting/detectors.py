"""Vessel detection strategies the experiment compares.

Every strategy returns the undilated boolean vessel mask of a volume. Dilation
and inpainting are applied afterwards, identically for every strategy.

* ``axial``: the pipeline's detector, slice by slice
  (``inpaint_vessels.get_slice_vessel_mask``).
* ``frangi3d``: the same recipe on the whole volume instead of one slice at a
  time, using the pipeline's Frangi parameters unchanged. ``FRANGI_SIGMAS`` is
  passed as ``sigmas``, as the pipeline does, which makes it three scales.
* ``frangi3d_range``: ``frangi3d``, but ``FRANGI_SIGMAS`` is read as
  ``(start, stop, step)`` and expanded with ``np.arange``, as the original
  paper specifies the scales.
"""

from collections.abc import Callable, Iterable
from typing import Literal, cast

import numpy as np
from joblib import Parallel, delayed
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from skimage.filters import frangi
from sklearn.cluster import KMeans

from microbleednet.constants import (
    CLUSTERER_N_CLUSTERS,
    CLUSTERER_RANDOM_STATE,
    FRANGI_ALPHA,
    FRANGI_BETA,
    FRANGI_BLACK_RIDGES,
    FRANGI_SIGMAS,
)
from microbleednet.core.transforms import inpaint_vessels

Strategy = Literal["axial", "frangi3d", "frangi3d_range"]
# Rescales the 3D vesselness before clustering; see ``frangi3d_vessel_mask``.
VESSELNESS_PERCENTILE = 99.9


def axial_vessel_mask(volume: np.ndarray) -> np.ndarray:
    """Detect vessels slice by slice, exactly like the pipeline (before dilation)."""
    slices = cast(
        list[np.ndarray],
        Parallel(n_jobs=-1)(
            delayed(inpaint_vessels.get_slice_vessel_mask)(volume[:, :, index])
            for index in range(volume.shape[2])
        ),
    )
    return np.stack(slices, axis=2).astype(bool)


def tube_linearity(volume: np.ndarray) -> np.ndarray:
    """Measure tube-like structure from the 3D structure-tensor eigenvalues.

    The pipeline's 2D measure ``|l1 - l2| / 2`` is large for a line (one strong
    gradient direction) and zero for a blob (two equal ones). In 3D a tube has
    two equal strong directions and one weak, so the analogue is the gap between
    the two smaller eigenvalues: large for a tube, zero for a blob (three equal)
    and for a plate (one strong, two weak).
    """
    eigenvalues = structure_tensor_eigenvalues(structure_tensor(volume))
    return np.absolute(eigenvalues[1] - eigenvalues[2]) / 2


def frangi3d_vessel_mask(
    volume: np.ndarray, sigmas: Iterable[float] = FRANGI_SIGMAS
) -> np.ndarray:
    """Detect vessels in 3D with the pipeline's Frangi parameters.

    Follows the pipeline's recipe: Frangi vesselness and a linearity measure,
    both restricted to the brain (voxels above zero), are clustered into two
    groups with KMeans, and the smaller group is the vessels. Unlike the axial
    detector there is no shape filter on the resulting regions, because that
    filter rejects round components, and a vessel running through the slice
    plane looks round in a single slice.

    Frangi is called with ``sigmas`` (by default the pipeline's, used as given)
    and the pipeline's ``alpha``, ``beta`` and ``black_ridges``. In 3D ``alpha``
    takes effect (it is unused in 2D) and, at the pipeline's value, shrinks
    responses to about 1e-3, which would make this feature irrelevant next to
    the 0 to 1 linearity. The maximum cannot be used to rescale it, because
    degenerate voxels (eigenvalues clipped to 1e-10 inside Frangi) can reach 1
    on their own. The vesselness is therefore divided by its
    ``VESSELNESS_PERCENTILE`` percentile within the brain and clipped to 1.
    """
    brain = volume > 0
    if not brain.any() or volume.min() == volume.max():
        return np.zeros(volume.shape, dtype=bool)

    vesselness = (
        frangi(
            volume,
            sigmas=sigmas,  # pyright: ignore[reportArgumentType]
            beta=FRANGI_BETA,
            alpha=FRANGI_ALPHA,
            black_ridges=FRANGI_BLACK_RIDGES,
        )
        * brain
    )
    scale = np.percentile(vesselness[brain], VESSELNESS_PERCENTILE)
    if scale > 0:
        vesselness = np.minimum(vesselness / scale, 1.0)

    linearity = tube_linearity(volume) * brain
    linearity -= np.min(linearity)
    if np.max(linearity) > 0:
        linearity = linearity / np.max(linearity)

    features = np.stack([vesselness.ravel(), linearity.ravel()], axis=1)
    clusters = (
        KMeans(n_clusters=CLUSTERER_N_CLUSTERS, random_state=CLUSTERER_RANDOM_STATE)
        .fit(features)
        .labels_
    )
    vessel_label = 1 if (clusters == 1).sum() < (clusters == 0).sum() else 0
    return np.reshape(clusters == vessel_label, volume.shape)


def frangi3d_range_vessel_mask(volume: np.ndarray) -> np.ndarray:
    """``frangi3d`` with ``FRANGI_SIGMAS`` read as ``(start, stop, step)``."""
    return frangi3d_vessel_mask(volume, sigmas=frangi_scale_range())


def frangi_scale_range() -> np.ndarray:
    """Expand ``FRANGI_SIGMAS`` as a range: 0.5, 0.7, 0.9, 1.1 for (0.5, 1.2, 0.2)."""
    start, stop, step = FRANGI_SIGMAS
    return np.arange(start, stop, step)


DETECTORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "axial": axial_vessel_mask,
    "frangi3d": frangi3d_vessel_mask,
    "frangi3d_range": frangi3d_range_vessel_mask,
}
