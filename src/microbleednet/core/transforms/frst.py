
import math

import numpy as np
import torch
import torchvision.transforms.functional as F
from scipy.ndimage import binary_erosion

from ...constants import (
    FRST_ALPHA,
    FRST_EDGE_MARGIN,
    FRST_FACTOR_STD,
    FRST_RADII,
)


def apply(
    volume: np.ndarray,
    brain_mask: np.ndarray,
) -> np.ndarray:
    """
    Apply slice-wise 2D FRST to a 3D volume.

    Gradients and voting are computed independently in each XY slice,
    while projection normalization and gradient thresholding use
    the entire volume.

    Input and output have shape (H, W, D).
    """
    volume_tensor = torch.from_numpy(
        np.asarray(volume)
    ).float()

    height, width, _ = volume_tensor.shape

    # Arrange slices as (D, H, W).
    slices = volume_tensor.permute(2, 0, 1)
    slice_count = slices.shape[0]

    # Exclude the brain boundary from voting.
    interior = binary_erosion(
        brain_mask > 0,
        structure=np.ones((3, 3, 1), dtype=bool),
        iterations=FRST_EDGE_MARGIN,
    )

    interior_t = torch.from_numpy(interior).permute(2, 0, 1)

    # Calculate 2D gradients independently for every slice.
    grad_y, grad_x = torch.gradient(slices, dim=(1, 2))
    g_norm = torch.sqrt(grad_x**2 + grad_y**2)

    # Gradient threshold: reject the weakest 20% of
    # nonzero gradients inside the brain, across the volume.
    interior_grads = g_norm[interior_t]
    nonzero_grads = interior_grads[interior_grads > 0]

    if nonzero_grads.numel() == 0:
        return np.zeros_like(volume, dtype=np.float32)

    grad_thresh = torch.quantile(nonzero_grads, 0.20)

    significant = (
        interior_t
        & (g_norm >= grad_thresh)
    )

    grad_x_significant = grad_x[significant]
    grad_y_significant = grad_y[significant]
    g_norm_significant = g_norm[significant]

    coords = torch.nonzero(significant)
    nn = coords[:, 0]
    yy = coords[:, 1]
    xx = coords[:, 2]

    offset = int(math.ceil(max(FRST_RADII)))
    out_height = height + 2 * offset
    out_width = width + 2 * offset

    output = torch.zeros(
        (slice_count, out_height, out_width), dtype=volume_tensor.dtype
    )

    for radius in FRST_RADII:
        orientation = torch.zeros_like(output)
        magnitude = torch.zeros_like(output)

        # Positive votes only: bright radially symmetric structures.
        gp_y = torch.round(
            (grad_y_significant / g_norm_significant) * radius
        ).long()

        gp_x = torch.round(
            (grad_x_significant / g_norm_significant) * radius
        ).long()

        pos_y = yy + gp_y + offset
        pos_x = xx + gp_x + offset

        idx_bright = nn * (out_height * out_width) + pos_y * out_width + pos_x
        orientation.view(-1).scatter_add_(
            0, idx_bright, torch.ones_like(idx_bright, dtype=volume_tensor.dtype)
        )
        magnitude.view(-1).scatter_add_(0, idx_bright, g_norm_significant)

        orientation = torch.abs(orientation)
        orientation = normalize_tensor_volume(orientation)

        magnitude = torch.abs(magnitude)
        magnitude = normalize_tensor_volume(magnitude)

        response = (orientation**FRST_ALPHA) * magnitude

        sigma = radius * FRST_FACTOR_STD
        rad_int = int(4.0 * sigma + 0.5)
        kernel_size = 2 * rad_int + 1

        response = response.unsqueeze(1)
        response = F.gaussian_blur(
            response, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma]
        )
        output += response.squeeze(1)

    output = output / len(FRST_RADII)

    output = output[:, offset:-offset, offset:-offset]

    result = output.permute(1, 2, 0).numpy()
    return result


def normalize_tensor_volume(tensor: torch.Tensor) -> torch.Tensor:
    """Scale a whole volume to the zero-to-one range by its maximum."""
    # One scale for the whole volume keeps a lesion's response independent of
    # whatever else shares its slice.
    maximum = tensor.max()
    if maximum <= 0:
        return tensor
    return tensor / maximum
