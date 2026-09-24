import numpy as np
from skimage.measure import regionprops

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
    padding = [(0, max(patch_size - size, 0)) for size in volume.shape]
    padded = np.pad(volume, padding, mode="constant", constant_values=0)
    starts = [
        list(range(0, size - patch_size + 1, patch_size))
        for size in padded.shape
    ]
    for axis, size in enumerate(padded.shape):
        final_start = size - patch_size
        if starts[axis][-1] != final_start:
            starts[axis].append(final_start)

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
