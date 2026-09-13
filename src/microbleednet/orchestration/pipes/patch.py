"""Create materialized training patches from preprocessed subjects."""

from typing import cast

import numpy as np

from ...core import io
from ...core import utils as core_utils
from ...core.common.models import CandidateDetector
from ...core.datamodels import ExtractedPatches, PatchRecord
from ...core.engines import processor as core_processor
from ...core.io import save_array_atomic
from ...core.transforms import patch as patch_transforms
from ..configs import (
    BasePatchConfig,
    NonOverlappingPatchConfig,
    TargetCenteredPatchConfig,
)


def execute(
    config: BasePatchConfig,
) -> list[PatchRecord]:
    if isinstance(config, NonOverlappingPatchConfig):
        extract = NonOverlappingExtractor(config.patch_size)
    elif isinstance(config, TargetCenteredPatchConfig):
        extract = TargetCenteredExtractor(
            detector=cast(CandidateDetector, config.detector),
            threshold=config.probability_threshold,
            patch_size=config.patch_size,
        )
    else:
        raise TypeError(f"unsupported patch configuration: {type(config).__name__}")

    records: list[PatchRecord] = []
    config.patch_dir.mkdir(parents=True, exist_ok=True)

    for subject in config.subjects:
        subject_id = subject.subject_id
        volume = io.nifti_to_numpy(io.load_volume(subject.volume_path))
        mask = io.nifti_to_numpy(io.load_volume(subject.mask_path))

        extracted = extract(volume, mask)
        if extracted.volumes.size == 0:
            continue

        volume_path = config.patch_dir / f"volumes_{subject_id}.npy"
        save_array_atomic(extracted.volumes, volume_path)
        
        mask_path = config.patch_dir / f"masks_{subject_id}.npy"
        save_array_atomic(extracted.masks, mask_path)

        for index, mask_array in enumerate(extracted.masks):
            for replica in range(config.augmentation_factor):
                records.append(
                    PatchRecord(
                        volume_path=str(volume_path.resolve()),
                        mask_path=str(mask_path.resolve()),
                        patch_index=index,
                        has_microbleed=bool(np.any(mask_array > 0)),
                        augmented=replica > 0,
                    )
                )

    return records


class NonOverlappingExtractor:
    """Extract aligned non-overlapping patches from a volume and its mask."""

    def __init__(self, patch_size: int):
        self.patch_size = patch_size

    def __call__(self, volume: np.ndarray, mask: np.ndarray) -> ExtractedPatches:
        volume_patches = patch_transforms.get_nonoverlapping_patches(
            volume, self.patch_size
        )
        mask_patches = patch_transforms.get_nonoverlapping_patches(
            mask, self.patch_size
        )
        return ExtractedPatches(
            volumes=np.stack(volume_patches),
            masks=np.stack(mask_patches),
        )


class TargetCenteredExtractor:
    """Center one patch on each detector candidate."""

    def __init__(
        self,
        detector: CandidateDetector,
        threshold: float,
        patch_size: int,
    ):
        self.detector = detector
        self.threshold = threshold
        self.patch_size = patch_size

    def __call__(self, volume: np.ndarray, mask: np.ndarray) -> ExtractedPatches:
        logits = core_processor.infer(self.detector, volume)
        output = core_utils.microbleed_probability(logits)
        candidate_mask = output > self.threshold
        centers = patch_transforms.get_target_centers(candidate_mask)
        volume_patches = patch_transforms.extract_centered_patches(
            volume, centers, self.patch_size
        )
        mask_patches = patch_transforms.extract_centered_patches(
            mask, centers, self.patch_size
        )
        if not volume_patches:
            empty_shape = (0, self.patch_size, self.patch_size, self.patch_size)
            return ExtractedPatches(
                volumes=np.empty(empty_shape),
                masks=np.empty(empty_shape),
            )
        return ExtractedPatches(
            volumes=np.stack(volume_patches),
            masks=np.stack(mask_patches),
        )
