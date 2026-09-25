import logging
from pathlib import Path
from typing import cast

import numpy as np

from ...constants import COMPONENT_CONNECTIVITY
from ...core import io, utils
from ...core.common.models import CandidateDetector
from ...core.datamodels import ExtractedPatches, PatchRecord
from ...core.engines import inference as core_inference
from ...core.io import save_array
from ...core.transforms import patch as patch_transforms
from ...errors import ApplicationError
from ...progress import progress
from ..configs import (
    BasePatchConfig,
    NonOverlappingPatchConfig,
    TargetCenteredPatchConfig,
)
from ..manifests import ManifestStatus, PatchManifest
from ..utils import resolve_path_string

logger = logging.getLogger(__name__)


def manifest_matches_config(
    manifest: PatchManifest, config: BasePatchConfig
) -> bool:
    """Check whether a complete patch manifest is reusable for a configuration."""
    probability_threshold = (
        config.probability_threshold
        if isinstance(config, TargetCenteredPatchConfig)
        else None
    )
    record_paths = {
        path
        for record in manifest.records
        for path in (record.volume_path, record.mask_path, record.frst_path)
    }
    return (
        manifest.status is ManifestStatus.COMPLETE
        and manifest.stage == config.stage
        and manifest.split == config.split
        and manifest.subject_ids
        == [subject.subject_id for subject in config.subjects]
        and manifest.patch_size == config.patch_size
        and manifest.augmentation_factor == config.augmentation_factor
        and manifest.probability_threshold == probability_threshold
        and all(Path(path).is_file() for path in record_paths)
    )


def execute(
    config: BasePatchConfig,
) -> None:
    """Extract materialized patches and write their completed manifest."""
    if isinstance(config, NonOverlappingPatchConfig):
        extract = NonOverlappingExtractor(config.patch_size)
    elif isinstance(config, TargetCenteredPatchConfig):
        extract = TargetCenteredExtractor(
            detector=config.detector,
            threshold=config.probability_threshold,
            patch_size=config.patch_size,
        )
    else:
        raise TypeError(
            "Patch extraction requires a supported patch configuration, "
            f"received {type(config).__name__}"
        )

    records: list[PatchRecord] = []
    work_items = [
        (subject, variant_index, variant)
        for subject in config.subjects
        for variant_index, variant in enumerate(
            subject.variants[: config.augmentation_factor]
        )
    ]
    logger.info(
        "Extracting %s %s patches for %d subject-variant items",
        config.stage,
        config.split,
        len(work_items),
    )
    for subject, variant_index, variant in progress.track(
        work_items,
        description=f"Extracting {config.stage} {config.split} patches",
    ):
        subject_id = subject.subject_id
        volume = io.nifti_to_numpy(io.load_volume(variant.volume_path))
        mask = io.nifti_to_numpy(io.load_volume(cast(str, variant.mask_path)))
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
        save_array(extracted.volumes, volume_path)
        save_array(extracted.masks, mask_path)
        save_array(extracted.frst, frst_path)

        records.extend(
            PatchRecord(
                volume_path=resolve_path_string(volume_path),
                mask_path=resolve_path_string(mask_path),
                frst_path=resolve_path_string(frst_path),
                patch_index=index,
                has_microbleed=bool(np.any(mask_array > 0)),
            )
            for index, mask_array in enumerate(extracted.masks)
        )

    if config.subjects and not records:
        raise ApplicationError(
            category="Input data",
            summary="Patch extraction produced no records",
            cause=(
                f"No {config.stage} {config.split} patches were produced for "
                f"{len(config.subjects)} subjects"
            ),
            fix=(
                "Check subject masks, patch size, augmentation settings, and split "
                "contents"
            ),
        )

    manifest_path = config.experiment_layout.patch_manifest_path(
        config.stage, config.split
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
    ).write(manifest_path)
    logger.info(
        "Patch extraction complete: %d patches written to %s",
        len(records),
        manifest_path,
    )


class NonOverlappingExtractor:
    """Extract aligned non-overlapping patches from a volume and its mask."""

    def __init__(self, patch_size: int):
        """Store the cubic patch size used for extraction."""
        self.patch_size = patch_size

    def __call__(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        frst: np.ndarray,
    ) -> ExtractedPatches:
        """Split aligned volume, mask, and FRST arrays into patch stacks."""
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
        """Store the detector and candidate-centered extraction settings."""
        self.detector = detector
        self.threshold = threshold
        self.patch_size = patch_size

    def __call__(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        frst: np.ndarray,
    ) -> ExtractedPatches:
        """Extract aligned patches centered on thresholded detector candidates."""
        probability_map = core_inference.infer_detector(
            self.detector, volume, frst
        )
        candidate_labels = utils.label_components(
            probability_map > self.threshold, COMPONENT_CONNECTIVITY
        )
        centers = patch_transforms.get_target_centers(candidate_labels)
        if not centers:
            empty_shape = (0, self.patch_size, self.patch_size, self.patch_size)
            return ExtractedPatches(
                volumes=np.empty(empty_shape, dtype=volume.dtype),
                masks=np.empty(empty_shape, dtype=mask.dtype),
                frst=np.empty(empty_shape, dtype=frst.dtype),
            )
        volume_patches = patch_transforms.extract_centered_patches(
            volume, centers, self.patch_size
        )
        mask_patches = patch_transforms.extract_centered_patches(
            mask, centers, self.patch_size
        )
        frst_patches = patch_transforms.extract_centered_patches(
            frst, centers, self.patch_size
        )
        return ExtractedPatches(
            volumes=np.stack(volume_patches),
            masks=np.stack(mask_patches),
            frst=np.stack(frst_patches),
        )
