import numpy as np
from skimage.measure import regionprops

from ..datamodels import BoundingBox, Shape3D
from . import volume_ops


def get_target_centers(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Return rounded centers of connected labeled target regions."""
    return [
        (
            int(round(region.centroid[0])),
            int(round(region.centroid[1])),
            int(round(region.centroid[2])),
        )
        for region in regionprops(labels)
    ]


def extract_centered_patches(
    volume: np.ndarray,
    centers: list[tuple[int, int, int]],
    patch_size: int,
) -> list[np.ndarray]:
    """Extract one cubic target-centered patch around each center."""
    half = patch_size // 2
    padded = np.pad(volume, half, mode="constant", constant_values=0)
    patches = []
    for center in centers:
        region = tuple(slice(axis, axis + patch_size) for axis in center)
        patches.append(padded[region])
    return patches


def get_nonoverlapping_patches(
    volume: np.ndarray,
    patch_size: int,
) -> list[np.ndarray]:
    """Extract non-overlapping cubic patches."""
    padding = [(0, -size % patch_size) for size in volume.shape]
    padded = np.pad(volume, padding, mode="constant", constant_values=0)
    starts = [
        list(range(0, size - patch_size + 1, patch_size))
        for size in padded.shape
    ]

    patches = []
    for start_z in starts[2]:
        for start_y in starts[1]:
            for start_x in starts[0]:
                bounding_box = (
                    (start_x, start_x + patch_size),
                    (start_y, start_y + patch_size),
                    (start_z, start_z + patch_size),
                )
                patches.append(volume_ops.apply_bounding_box(padded, bounding_box))

    return patches


def get_overlapping_patch_bounding_boxes(
    shape: Shape3D,
    patch_size: int,
    overlap: int,
) -> list[BoundingBox]:
    """Return overlapping cubic bounds that fully cover a volume."""
    axis_starts = [
        get_axis_starts(size, patch_size, overlap) for size in shape
    ]
    return [
        (
            (start_0, min(start_0 + patch_size, shape[0])),
            (start_1, min(start_1 + patch_size, shape[1])),
            (start_2, min(start_2 + patch_size, shape[2])),
        )
        for start_2 in axis_starts[2]
        for start_1 in axis_starts[1]
        for start_0 in axis_starts[0]
    ]


def get_axis_starts(size: int, patch_size: int, overlap: int) -> list[int]:
    """Return patch starts that cover one axis, including its trailing edge."""
    if size <= patch_size:
        return [0]

    stride = patch_size - overlap
    starts = list(range(0, size - patch_size + 1, stride))
    final_start = size - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts
