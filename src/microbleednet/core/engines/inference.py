import numpy as np
import torch.nn as nn
from skimage.measure import regionprops

from ...constants import COMPONENT_CONNECTIVITY
from .. import utils
from ..transforms import patch as patch_transforms


def infer_detector(
    detector: nn.Module,
    volume: np.ndarray,
    frst: np.ndarray,
) -> np.ndarray:
    """Return the detector's full-volume microbleed probability map."""
    model_input = utils.stack_volume_and_frst(volume, frst)
    logits = utils.predict_logits(detector, model_input)
    return utils.microbleed_probability(logits)


def infer_discriminator(
    student: nn.Module,
    volume: np.ndarray,
    frst: np.ndarray,
    detector_probability: np.ndarray,
    detector_threshold: float,
    patch_size: int,
    discriminator_threshold: float,
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
        logits = utils.predict_logits(student, patch)
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

