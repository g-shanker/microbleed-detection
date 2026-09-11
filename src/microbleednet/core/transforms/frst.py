import math

import torch
import torchvision.transforms.functional as F

# Fast Radial Symmetry Transform structural parameters.
FRST_RADII = [2, 3, 4, 6]
FRST_ALPHA = 2
FRST_FACTOR_STD = 0.1


def apply(
    volumes: torch.Tensor,
) -> torch.Tensor:
    """
    Batched 3D FRST on GPU.
    Input: volumes (Batch, 1, H, W, D)
    Output: frst_volumes (Batch, 1, H, W, D)
    """
    batch_size, _, height, width, depth = volumes.shape

    slices = volumes.permute(0, 4, 2, 3, 1).reshape(-1, height, width)
    slice_count = slices.shape[0]

    grad_y, grad_x = torch.gradient(slices, dim=(1, 2))
    g_norm = torch.sqrt(grad_x**2 + grad_y**2)

    significant = g_norm > 0
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
        (slice_count, out_height, out_width), device=volumes.device, dtype=volumes.dtype
    )

    for radius in FRST_RADII:
        orientation = torch.zeros(
            (slice_count, out_height, out_width),
            device=volumes.device,
            dtype=volumes.dtype,
        )
        magnitude = torch.zeros(
            (slice_count, out_height, out_width),
            device=volumes.device,
            dtype=volumes.dtype,
        )
        gp_y = torch.round((grad_y_significant / g_norm_significant) * radius).long()
        gp_x = torch.round((grad_x_significant / g_norm_significant) * radius).long()

        pos_y = yy + gp_y + offset
        pos_x = xx + gp_x + offset

        idx_bright = nn * (out_height * out_width) + pos_y * out_width + pos_x
        orientation.view(-1).scatter_add_(
            0, idx_bright, torch.ones_like(idx_bright, dtype=volumes.dtype)
        )
        magnitude.view(-1).scatter_add_(0, idx_bright, g_norm_significant)

        orientation = torch.abs(orientation)
        orientation = normalize_tensor_slicewise(orientation)

        magnitude = torch.abs(magnitude)
        magnitude = normalize_tensor_slicewise(magnitude)

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

    # Reshape back to (Batch, 1, Height, Width, Depth)
    output = (
        output.view(batch_size, depth, height, width)
        .unsqueeze(1)
        .permute(0, 1, 3, 4, 2)
    )
    return output


def normalize_tensor_slicewise(tensor: torch.Tensor) -> torch.Tensor:
    # Assumes tensor shape is (N, H, W)
    t_min = tensor.amin(dim=(1, 2), keepdim=True)
    t_max = tensor.amax(dim=(1, 2), keepdim=True)

    tensor = tensor - t_min
    t_range = t_max - t_min

    # Divide only where the range is > 0 to avoid division by zero
    tensor = torch.where(t_range > 0, tensor / t_range, tensor)

    return tensor
