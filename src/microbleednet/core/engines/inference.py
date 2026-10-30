import numpy as np
import torch.nn as nn
from skimage.measure import regionprops

from ...constants import (
    COMPONENT_CONNECTIVITY,
    DETECTOR_INFERENCE_TILE_OVERLAP,
    DETECTOR_INFERENCE_TILE_SIZE,
)
from .. import utils
from ..transforms import patch as patch_transforms
from ..transforms import volume_ops


def infer_detector(
    detector: nn.Module,
    volume: np.ndarray,
    frst: np.ndarray,
    use_amp: bool,
) -> np.ndarray:
    """Return a detector probability map using bounded-memory tiled inference."""
    spatial_shape = volume.shape

    if all(size <= DETECTOR_INFERENCE_TILE_SIZE for size in spatial_shape):
        model_input = utils.stack_volume_and_frst(volume, frst)
        logits = utils.predict_logits(detector, model_input, use_amp)

        return utils.microbleed_probability(logits)

    probability_sum = np.zeros(spatial_shape, dtype=np.float32)
    prediction_count = np.zeros(spatial_shape, dtype=np.uint8)
    bounding_boxes = patch_transforms.get_overlapping_patch_bounding_boxes(
        spatial_shape,
        DETECTOR_INFERENCE_TILE_SIZE,
        DETECTOR_INFERENCE_TILE_OVERLAP,
    )
    for bounding_box in bounding_boxes:
        model_input = utils.stack_volume_and_frst(
            volume_ops.apply_bounding_box(volume, bounding_box),
            volume_ops.apply_bounding_box(frst, bounding_box),
        )
        logits = utils.predict_logits(detector, model_input, use_amp)
        probability = utils.microbleed_probability(logits)
        volume_ops.add_to_bounding_box(
            probability_sum, probability, bounding_box
        )
        volume_ops.add_to_bounding_box(prediction_count, 1, bounding_box)

    return probability_sum / prediction_count


def infer_discriminator(
    student: nn.Module,
    volume: np.ndarray,
    frst: np.ndarray,
    detector_probability: np.ndarray,
    detector_threshold: float,
    patch_size: int,
    discriminator_threshold: float,
    use_amp: bool,
) -> np.ndarray:
    """Threshold detector candidates, classify patches, and return retained labels."""
    candidate_labels = utils.label_components(
        detector_probability > detector_threshold, COMPONENT_CONNECTIVITY
    )
    centers = patch_transforms.get_target_centers(candidate_labels)
    if not centers:
        return np.zeros_like(candidate_labels, dtype=np.uint8)

    volume_patches = patch_transforms.extract_centered_patches(
        volume, centers, patch_size
    )
    frst_patches = patch_transforms.extract_centered_patches(
        frst, centers, patch_size
    )
    retained = []
    for volume_patch, frst_patch in zip(
        volume_patches, frst_patches, strict=True
    ):
        patch = utils.stack_volume_and_frst(volume_patch, frst_patch)
        logits = utils.predict_logits(student, patch, use_amp)
        probability = utils.microbleed_probability(logits)
        retained.append(
            bool(probability >= discriminator_threshold)
        )

    output = np.zeros_like(candidate_labels, dtype=np.uint8)
    candidates = regionprops(candidate_labels)
    for candidate, keep in zip(candidates, retained, strict=True):
        if keep:
            output[candidate_labels == candidate.label] = 1
    return output

