"""Create materialized training patches from preprocessed subjects."""

from typing import cast

import numpy as np

from ...core import io, utils
from ...core.common.models import CandidateDetector
from ...core.datamodels import ExtractedPatches, PatchRecord
from ...core.engines import inference as core_inference
from ...core.io import save_array_atomic
from ...core.transforms import patch as patch_transforms
from ..configs import (
    BasePatchConfig,
    NonOverlappingPatchConfig,
    TargetCenteredPatchConfig,
)
from ..manifests import ManifestStatus, PatchManifest


def execute(
    config: BasePatchConfig,
) -> None:
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
    patch_dir = config.experiment_layout.patch_dir_path(config.stage, config.split)
    patch_dir.mkdir(parents=True, exist_ok=True)

    for subject in config.subjects:
        subject_id = subject.subject_id
        variants = subject.variants[: config.augmentation_factor]

        for variant_index, variant in enumerate(variants):
            volume = io.nifti_to_numpy(io.load_volume(variant.volume_path))
            mask = io.nifti_to_numpy(io.load_volume(variant.mask_path))
            frst = io.nifti_to_numpy(io.load_volume(variant.frst_path))
            extracted = extract(volume, mask, frst)
            if extracted.volumes.size == 0:
                continue

            volume_path = config.experiment_layout.patch_volume_path(
                config.stage, config.split, subject_id, variant_index
            )
            mask_path = config.experiment_layout.patch_mask_path(
                config.stage, config.split, subject_id, variant_index
            )
            frst_path = config.experiment_layout.patch_frst_path(
                config.stage, config.split, subject_id, variant_index
            )
            save_array_atomic(extracted.volumes, volume_path)
            save_array_atomic(extracted.masks, mask_path)
            save_array_atomic(extracted.frst, frst_path)

            records.extend(
                PatchRecord(
                    volume_path=str(volume_path.resolve()),
                    mask_path=str(mask_path.resolve()),
                    frst_path=str(frst_path.resolve()),
                    patch_index=index,
                    has_microbleed=bool(np.any(mask_array > 0)),
                )
                for index, mask_array in enumerate(extracted.masks)
            )

    PatchManifest(
        status=ManifestStatus.COMPLETE,
        stage=config.stage,
        split=config.split,
        subject_ids=[subject.subject_id for subject in config.subjects],
        patch_size=config.patch_size,
        augmentation_factor=config.augmentation_factor,
        probability_threshold=(
            config.probability_threshold
            if isinstance(config, TargetCenteredPatchConfig)
            else None
        ),
        records=records,
    ).write(config.experiment_layout.patch_manifest_path(config.stage, config.split))


class NonOverlappingExtractor:
    """Extract aligned non-overlapping patches from a volume and its mask."""

    def __init__(self, patch_size: int):
        self.patch_size = patch_size

    def __call__(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        frst: np.ndarray,
    ) -> ExtractedPatches:
        volume_patches = patch_transforms.get_nonoverlapping_patches(
            volume, self.patch_size
        )
        mask_patches = patch_transforms.get_nonoverlapping_patches(
            mask, self.patch_size
        )
        frst_patches = patch_transforms.get_nonoverlapping_patches(
            frst, self.patch_size
        )
        return ExtractedPatches(
            volumes=np.stack(volume_patches),
            masks=np.stack(mask_patches),
            frst=np.stack(frst_patches),
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

    def __call__(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        frst: np.ndarray,
    ) -> ExtractedPatches:
        probability_map = core_inference.infer_detector(
            self.detector, volume, frst
        )
        candidate_labels = utils.label_components(
            probability_map > self.threshold, utils.COMPONENT_CONNECTIVITY
        )
        centers = patch_transforms.get_target_centers(candidate_labels)
        volume_patches = patch_transforms.extract_centered_patches(
            volume, centers, self.patch_size
        )
        mask_patches = patch_transforms.extract_centered_patches(
            mask, centers, self.patch_size
        )
        frst_patches = patch_transforms.extract_centered_patches(
            frst, centers, self.patch_size
        )
        if not centers:
            empty_shape = (0, self.patch_size, self.patch_size, self.patch_size)
            return ExtractedPatches(
                volumes=np.empty(empty_shape),
                masks=np.empty(empty_shape),
                frst=np.empty(empty_shape),
            )
        return ExtractedPatches(
            volumes=np.stack(volume_patches),
            masks=np.stack(mask_patches),
            frst=np.stack(frst_patches),
        )
