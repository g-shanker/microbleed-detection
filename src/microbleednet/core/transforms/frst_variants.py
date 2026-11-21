"""Configurable FRST used only to compare variants of the pipeline transform.

Experimental. ``BASELINE`` reproduces ``frst.apply`` exactly; the other
settings change one aspect of it. See EXPERIMENTAL_CHANGES.md for removal.
"""

import math
from dataclasses import dataclass, replace
from typing import Any, Literal, Sequence

import numpy as np
import torch
import torchvision.transforms.functional as F
from scipy.ndimage import binary_erosion, gaussian_filter, gaussian_filter1d
from skimage.filters import frangi

from ...constants import (
    FRANGI_ALPHA,
    FRANGI_BETA,
    FRANGI_BLACK_RIDGES,
    FRANGI_SIGMAS,
    FRST_ALPHA,
    FRST_EDGE_MARGIN,
    FRST_FACTOR_STD,
    FRST_RADII,
)
from ...errors import ApplicationError

# Mirrors the literal gradient quantile in frst.apply.
BASELINE_GRADIENT_QUANTILE = 0.2
# Loy and Zelinsky's orientation caps for radius 1 and larger radii.
SMALL_RADIUS_CAP = 9.9
RADIUS_CAP = 8.0
CONTRAST_SIGMA_MM = 2.0
CONTRAST_EPSILON = 1e-6
VESSEL_SCALE_PERCENTILE = 99
MM_RADII = (1.0, 1.5, 2.0, 3.0)


@dataclass(frozen=True)
class FrstSettings:
    """Every FRST choice the comparison varies."""

    radii: tuple[float, ...] = tuple(float(radius) for radius in FRST_RADII)
    radii_unit: Literal["voxel", "mm"] = "voxel"
    alpha: float = FRST_ALPHA
    sigma_factor: float = FRST_FACTOR_STD
    gradient_quantile: float = BASELINE_GRADIENT_QUANTILE
    normalization: Literal["volume_max", "fixed"] = "volume_max"
    zscore_output: bool = False
    presmooth_mm: float = 0.0
    contrast_norm: bool = False
    voting: Literal["2d", "3d"] = "2d"
    through_slice_mm: float = 0.0
    max_extent_mm: float = 0.0
    vessel_suppression: bool = False


BASELINE = FrstSettings()
VARIANT_OVERRIDES: dict[str, dict[str, Any]] = {
    "baseline": {},
    "normalization_fixed": {"normalization": "fixed"},
    "zscore_output": {"zscore_output": True},
    "sigma_0.1": {"sigma_factor": 0.1},
    "sigma_0.25": {"sigma_factor": 0.25},
    "gradient_q0.0": {"gradient_quantile": 0.0},
    "gradient_q0.4": {"gradient_quantile": 0.4},
    "radii_mm": {"radii": MM_RADII, "radii_unit": "mm"},
    "radii_small": {"radii": (1.0, 2.0, 3.0, 4.0)},
    "radii_no_largest": {"radii": (2.0, 3.0, 4.0)},
    "alpha_1": {"alpha": 1.0},
    "alpha_3": {"alpha": 3.0},
    "presmooth_0.5mm": {"presmooth_mm": 0.5},
    "contrast_norm": {"contrast_norm": True},
    "vessel_suppression": {"vessel_suppression": True},
    "end_on_5mm": {"max_extent_mm": 5.0},
    "2.5d_1mm": {"through_slice_mm": 1.0},
    "3d_mm": {"voting": "3d", "radii": MM_RADII, "radii_unit": "mm"},
}


def settings_for(names: Sequence[str]) -> FrstSettings:
    """Combine the overrides of named variants into one setting."""
    merged: dict[str, Any] = {}
    for name in names:
        if name not in VARIANT_OVERRIDES:
            raise ApplicationError(
                category="Configuration",
                summary=f"Unknown FRST variant '{name}'",
                fix=f"Choose from: {', '.join(VARIANT_OVERRIDES)}",
            )
        for key, value in VARIANT_OVERRIDES[name].items():
            if key in merged and merged[key] != value:
                raise ApplicationError(
                    category="Configuration",
                    summary="FRST variants set the same option differently",
                    cause=f"'{key}' is both {merged[key]!r} and {value!r}",
                    fix="Combine variants that change different options",
                )
            merged[key] = value
    return replace(BASELINE, **merged)


def compute(
    volume: np.ndarray,
    brain_mask: np.ndarray,
    spacing: Sequence[float],
    settings: FrstSettings = BASELINE,
    device: str = "cpu",
) -> np.ndarray:
    """Apply FRST with the given settings; input and output are (H, W, D)."""
    spacing_mm = np.asarray(spacing, dtype=float)
    three_d = settings.voting == "3d"
    axes = 3 if three_d else 2
    # Voxel radii keep the pipeline's unit-spacing geometry.
    geometry = spacing_mm if settings.radii_unit == "mm" else np.ones(3)

    image = np.asarray(volume, dtype=np.float32)
    if settings.presmooth_mm > 0:
        image = gaussian_filter(
            image, sigma=mm_sigma(settings.presmooth_mm, spacing_mm, three_d)
        )
    interior = binary_erosion(
        brain_mask > 0,
        structure=np.ones((3, 3, 1), dtype=bool),
        iterations=FRST_EDGE_MARGIN,
    )
    interior_tensor = torch.from_numpy(interior).to(device)

    tensor = torch.from_numpy(image).float().to(device)
    gradients = torch.gradient(
        tensor,
        spacing=tuple(float(value) for value in geometry[:axes]),
        dim=tuple(range(axes)),
    )
    g_norm = torch.sqrt(sum(gradient**2 for gradient in gradients))
    weight = g_norm
    if settings.contrast_norm:
        weight = g_norm / (
            torch.from_numpy(local_std(image, spacing_mm, three_d)).to(device)
            + CONTRAST_EPSILON
        )

    interior_weights = weight[interior_tensor]
    nonzero = interior_weights[interior_weights > 0]
    if nonzero.numel() == 0:
        return np.zeros(image.shape, dtype=np.float32)
    threshold = torch.quantile(nonzero, settings.gradient_quantile)
    significant = interior_tensor & (weight >= threshold)

    coords = torch.nonzero(significant)
    units = [gradient[significant] / g_norm[significant] for gradient in gradients]
    votes = weight[significant]
    largest = max(settings.radii)
    pads = [
        int(math.ceil(largest / geometry[axis])) if axis < axes else 0
        for axis in range(3)
    ]
    padded = tuple(size + 2 * pad for size, pad in zip(image.shape, pads))
    strides = (padded[1] * padded[2], padded[2], 1)

    output = torch.zeros(padded, dtype=torch.float32, device=device)
    for radius in settings.radii:
        index = torch.zeros(coords.shape[0], dtype=torch.long, device=device)
        for axis in range(3):
            position = coords[:, axis] + pads[axis]
            if axis < axes:
                position = position + torch.round(
                    units[axis] * radius / float(geometry[axis])
                ).long()
            index = index + position * strides[axis]
        orientation = torch.zeros(padded, dtype=torch.float32, device=device)
        magnitude = torch.zeros(padded, dtype=torch.float32, device=device)
        orientation.view(-1).scatter_add_(0, index, torch.ones_like(votes))
        magnitude.view(-1).scatter_add_(0, index, votes)

        if settings.normalization == "fixed":
            cap = orientation_cap(radius, geometry, three_d)
            orientation = torch.clamp(orientation, max=cap) / cap
            magnitude = magnitude / cap
        else:
            orientation = orientation / orientation.max()
            magnitude = magnitude / magnitude.max()

        response = (orientation**settings.alpha) * magnitude
        output += blur(response, settings.sigma_factor * radius, geometry, three_d)

    output = output / len(settings.radii)
    crop = tuple(slice(pad, size - pad) for pad, size in zip(pads, padded))
    result = output[crop].cpu().numpy()
    return postprocess(result, image, brain_mask > 0, spacing_mm, settings)


def mm_sigma(
    sigma_mm: float, spacing: np.ndarray, three_d: bool
) -> tuple[float, float, float]:
    """Convert a physical sigma to voxels, in-plane only for 2D voting."""
    return (
        sigma_mm / spacing[0],
        sigma_mm / spacing[1],
        sigma_mm / spacing[2] if three_d else 0.0,
    )


def local_std(image: np.ndarray, spacing: np.ndarray, three_d: bool) -> np.ndarray:
    """Return the local intensity standard deviation over about 2 mm."""
    sigma = mm_sigma(CONTRAST_SIGMA_MM, spacing, three_d)
    mean = gaussian_filter(image, sigma)
    mean_square = gaussian_filter(image * image, sigma)
    return np.sqrt(np.maximum(mean_square - mean * mean, 0.0)).astype(np.float32)


def orientation_cap(radius: float, geometry: np.ndarray, three_d: bool) -> float:
    """Return the Loy-Zelinsky orientation cap, scaled by slices spanned in 3D."""
    in_plane_radius = radius / float(min(geometry[0], geometry[1]))
    cap = SMALL_RADIUS_CAP if in_plane_radius <= 1 else RADIUS_CAP
    if three_d:
        cap *= 2 * math.floor(radius / float(geometry[2])) + 1
    return cap


def blur(
    response: torch.Tensor, sigma: float, geometry: np.ndarray, three_d: bool
) -> torch.Tensor:
    """Blur in-plane like frst.apply, then along slices for 3D voting."""
    sigma_h = sigma / float(geometry[0])
    sigma_w = sigma / float(geometry[1])
    slices = response.permute(2, 0, 1).unsqueeze(1)
    slices = F.gaussian_blur(
        slices,
        kernel_size=[kernel_size(sigma_w), kernel_size(sigma_h)],
        sigma=[sigma_w, sigma_h],
    )
    blurred = slices.squeeze(1).permute(1, 2, 0)
    if three_d:
        along_slices = gaussian_filter1d(
            blurred.cpu().numpy(), sigma / float(geometry[2]), axis=2
        )
        blurred = torch.from_numpy(along_slices).to(response.device)
    return blurred


def kernel_size(sigma: float) -> int:
    """Return the odd kernel size frst.apply uses for a sigma."""
    return 2 * int(4.0 * sigma + 0.5) + 1


def postprocess(
    result: np.ndarray,
    image: np.ndarray,
    brain: np.ndarray,
    spacing: np.ndarray,
    settings: FrstSettings,
) -> np.ndarray:
    """Apply the optional steps that act on the finished FRST map."""
    if settings.through_slice_mm > 0:
        result = gaussian_filter1d(
            result, settings.through_slice_mm / spacing[2], axis=2
        )
    if settings.max_extent_mm > 0:
        result = suppress_long_structures(result, settings.max_extent_mm, spacing[2])
    if settings.vessel_suppression:
        vessels = vesselness(image)
        scale = max(
            float(np.percentile(vessels[brain], VESSEL_SCALE_PERCENTILE)),
            CONTRAST_EPSILON,
        )
        result = result * (1.0 - np.clip(vessels / scale, 0.0, 1.0))
    if settings.zscore_output:
        values = result[brain]
        result = (result - values.mean()) / (values.std() or 1.0)
    return result.astype(np.float32)


def suppress_long_structures(
    result: np.ndarray, max_extent_mm: float, slice_spacing: float
) -> np.ndarray:
    """Remove responses that persist on both sides beyond a lesion's extent."""
    # A microbleed fades within its extent along the slice axis, so at least
    # one side is low; a vessel seen end-on stays high on both sides.
    shift = max(1, math.ceil(max_extent_mm / slice_spacing))
    above = np.zeros_like(result)
    below = np.zeros_like(result)
    above[:, :, :-shift] = result[:, :, shift:]
    below[:, :, shift:] = result[:, :, :-shift]
    return np.maximum(result - np.minimum(above, below), 0.0)


def vesselness(image: np.ndarray) -> np.ndarray:
    """Return slice-wise Frangi vesselness with the pipeline's settings."""
    slices = []
    for index in range(image.shape[2]):
        image_slice = image[:, :, index]
        if np.min(image_slice) == np.max(image_slice):
            slices.append(np.zeros_like(image_slice))
            continue
        slices.append(
            frangi(
                image_slice,
                sigmas=FRANGI_SIGMAS,  # pyright: ignore[reportArgumentType]
                beta=FRANGI_BETA,
                alpha=FRANGI_ALPHA,
                black_ridges=FRANGI_BLACK_RIDGES,
            )
        )
    return np.stack(slices, axis=2)
