import numpy as np

from ...constants import (
    AVAILABLE_TRANSFORMATIONS,
    BLUR_SIGMA_RANGE,
    NOISE_VARIANCE_RANGE,
    TRANSLATION_OFFSET_RANGE,
)
from . import volume_ops


def augment(
    volume: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply a random combination of the training augmentations."""
    rng = np.random.default_rng()
    count = int(rng.integers(1, len(AVAILABLE_TRANSFORMATIONS) + 1))
    transformations = rng.choice(
        AVAILABLE_TRANSFORMATIONS,
        size=count,
        replace=False,
    )

    transformed_volume = volume
    transformed_mask = mask

    for transformation in transformations:
        if transformation == "translate":
            low, high = TRANSLATION_OFFSET_RANGE
            offset_x = int(rng.integers(low, high + 1))
            offset_y = int(rng.integers(low, high + 1))
            transformed_volume = volume_ops.translate(
                transformed_volume, offset_x, offset_y
            )
            if transformed_mask is not None:
                transformed_mask = volume_ops.translate(
                    transformed_mask, offset_x, offset_y
                )
        elif transformation == "noise":
            variance = rng.uniform(*NOISE_VARIANCE_RANGE)
            noise = rng.normal(0, np.sqrt(variance), transformed_volume.shape)
            transformed_volume = volume_ops.add_noise(transformed_volume, noise)
        else:
            sigma = rng.uniform(*BLUR_SIGMA_RANGE)
            transformed_volume = volume_ops.blur(transformed_volume, sigma)

    return transformed_volume, transformed_mask
