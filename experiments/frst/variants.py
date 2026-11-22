"""A configurable FRST for comparing variants against the pipeline's FRST.

``FrstSpec()`` with default values reproduces ``microbleednet.core.transforms.frst``
(slice-wise 2D voting, per-slice normalization, every nonzero gradient votes).
Every other field is an experimental modification of that baseline.
"""

import math
from typing import Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.ndimage import binary_erosion, gaussian_filter, gaussian_filter1d

from microbleednet.constants import FRST_ALPHA, FRST_FACTOR_STD, FRST_RADII

Normalization = Literal["slice", "volume"]


class FrstSpec(BaseModel):
    """One FRST configuration; defaults match the pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["slice", "volume"] = Field(
        default="slice",
        description=(
            "'slice' votes independently in every axial slice. 'volume' votes in "
            "3D using physical voxel spacing, so thick slices vote less far."
        ),
    )
    radii: tuple[int, ...] = Field(
        default=FRST_RADII,
        min_length=1,
        description="Voting radii in in-plane voxels; the maps are averaged.",
    )
    alpha: float = Field(
        default=FRST_ALPHA,
        gt=0,
        description="Exponent that sharpens the orientation projection map.",
    )
    blur_factor: float = Field(
        default=FRST_FACTOR_STD,
        ge=0,
        description="Gaussian sigma of each radius map, as a fraction of the radius.",
    )
    gradient_quantile: float = Field(
        default=0.0,
        ge=0,
        lt=1,
        description=(
            "Fraction of the weakest nonzero gradients, taken volume-wide inside "
            "the brain, that do not vote. 0 lets every nonzero gradient vote."
        ),
    )
    edge_margin: int = Field(
        default=0,
        ge=0,
        description="In-plane voxels next to the brain boundary that do not vote.",
    )
    normalization: Normalization = Field(
        default="slice",
        description=(
            "Scale each voting map per axial slice (min-max) or once per volume "
            "(divide by the maximum)."
        ),
    )
    through_slice_blur_mm: float = Field(
        default=0.0,
        ge=0,
        description="Gaussian sigma in mm applied across slices to the final map.",
    )


BASELINE = FrstSpec()


def apply(
    volume: np.ndarray,
    brain_mask: np.ndarray,
    spacing: tuple[float, float, float],
    spec: FrstSpec,
) -> np.ndarray:
    """Compute the FRST of ``volume`` (shape ``(H, W, D)``) for one ``spec``."""
    data = np.asarray(volume, dtype=np.float32)
    shape = data.shape
    is_volume = spec.mode == "volume"
    axes = (0, 1, 2) if is_volume else (0, 1)
    in_plane = float(min(spacing[0], spacing[1]))
    # Voxels per in-plane voxel of radius along each axis.
    scale = (1.0, 1.0, in_plane / float(spacing[2]) if is_volume else 0.0)
    steps = tuple(float(spacing[axis]) if is_volume else 1.0 for axis in axes)

    gradients = np.stack(cast(list[np.ndarray], np.gradient(data, *steps, axis=axes)))
    magnitude = np.sqrt((gradients**2).sum(axis=0))
    significant = _voters(magnitude, brain_mask, spec)
    if not significant.any():
        return np.zeros(shape, dtype=np.float32)

    weights = magnitude[significant]
    unit = gradients[:, significant] / weights
    coords = np.nonzero(significant)

    pad = tuple(
        math.ceil(max(spec.radii) * scale[axis]) if axis in axes else 0
        for axis in range(3)
    )
    padded_shape = tuple(size + 2 * margin for size, margin in zip(shape, pad))
    size = int(np.prod(padded_shape))
    output = np.zeros(padded_shape, dtype=np.float32)

    for radius in spec.radii:
        positions = tuple(
            coords[axis]
            + (
                np.round(unit[axis] * radius * scale[axis]).astype(np.int64)
                if axis in axes
                else 0
            )
            + pad[axis]
            for axis in range(3)
        )
        flat = np.ravel_multi_index(positions, padded_shape)
        orientation = np.bincount(flat, minlength=size).reshape(padded_shape)
        projection = np.bincount(flat, weights=weights, minlength=size)
        response = _normalize(
            orientation.astype(np.float32), spec.normalization
        ) ** spec.alpha * _normalize(
            projection.reshape(padded_shape).astype(np.float32),
            spec.normalization,
        )

        sigma = radius * spec.blur_factor
        sigmas = (sigma, sigma, sigma * scale[2])
        output += gaussian_filter(response, sigmas, mode="mirror")

    output /= len(spec.radii)
    result = output[
        tuple(slice(margin, margin + extent) for margin, extent in zip(pad, shape))
    ]
    if spec.through_slice_blur_mm > 0:
        result = gaussian_filter1d(
            result,
            spec.through_slice_blur_mm / float(spacing[2]),
            axis=2,
            mode="nearest",
        )
    return np.asarray(result, dtype=np.float32)


def _voters(
    magnitude: np.ndarray, brain_mask: np.ndarray, spec: FrstSpec
) -> np.ndarray:
    """Return the voxels whose gradients vote."""
    interior: np.ndarray = np.asarray(brain_mask) > 0
    if spec.edge_margin > 0:
        interior = np.asarray(
            binary_erosion(
                interior,
                structure=np.ones((3, 3, 1), dtype=bool),
                iterations=spec.edge_margin,
            )
        )
    voters: np.ndarray = interior & (magnitude > 0)
    if spec.gradient_quantile > 0 and voters.any():
        threshold = np.quantile(magnitude[voters], spec.gradient_quantile)
        voters &= magnitude >= threshold
    return voters


def _normalize(tensor: np.ndarray, normalization: Normalization) -> np.ndarray:
    """Scale a voting map by slice (min-max) or by the volume maximum."""
    if normalization == "volume":
        maximum = tensor.max()
        return tensor / maximum if maximum > 0 else tensor
    minimum = tensor.min(axis=(0, 1), keepdims=True)
    value_range = tensor.max(axis=(0, 1), keepdims=True) - minimum
    shifted = tensor - minimum
    safe_range = np.where(value_range > 0, value_range, 1)
    return np.where(value_range > 0, shifted / safe_range, shifted)
