from typing import cast

import nibabel as nib
import numpy as np

from ...core import io
from ...core.datamodels import Modality
from ...core.engines import processor
from ...core.transforms import augmentations, frst
from .. import manifests
from ..configs import PreprocessConfig
from ..layouts import DatasetLayout
from ..manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    PreprocessedVariant,
    RawDatasetManifest,
)


def execute(config: PreprocessConfig) -> None:
    layout = DatasetLayout(dataset_dir=config.dataset_dir)
    raw_manifest = RawDatasetManifest.read(layout.raw_manifest_path())

    volumes_dir = layout.preprocessed_volumes_path()
    volumes_dir.mkdir(parents=True, exist_ok=True)

    masks_dir = layout.preprocessed_masks_path()
    masks_dir.mkdir(parents=True, exist_ok=True)

    frst_dir = layout.preprocessed_frst_path()
    frst_dir.mkdir(parents=True, exist_ok=True)

    source_modalities = {
        source.source_id: source.modality for source in raw_manifest.sources
    }
    preprocessed_subjects = []

    for subject in raw_manifest.subjects:
        subject_id = subject.subject_id

        raw_volume = io.load_volume(subject.volume_path)
        raw_mask = io.load_volume(subject.mask_path)

        modality = cast(Modality, source_modalities[subject.source_id])
        preprocess_result = processor.preprocess(raw_volume, raw_mask, modality)

        variants = []
        original_volume = preprocess_result.image
        original_mask = preprocess_result.mask
        for variant_index in range(config.augmentation_factor):
            volume = original_volume
            mask = original_mask
            if variant_index > 0:
                volume, mask = augmentations.augment(volume, mask)

            variant_volume = nib.Nifti1Image(volume, preprocess_result.affine)
            variant_mask = nib.Nifti1Image(mask, preprocess_result.affine)
            variant_frst = frst.apply(np.asarray(volume))
            variant_frst = nib.Nifti1Image(variant_frst, preprocess_result.affine)

            volume_path = layout.variant_volume_path(subject_id, variant_index)
            mask_path = layout.variant_mask_path(subject_id, variant_index)
            frst_path = layout.variant_frst_path(subject_id, variant_index)
            
            io.save_volume(variant_volume, volume_path)
            io.save_volume(variant_mask, mask_path)
            io.save_volume(variant_frst, frst_path)
            
            variants.append(
                PreprocessedVariant(
                    volume_path=str(volume_path.resolve()),
                    mask_path=str(mask_path.resolve()),
                    frst_path=str(frst_path.resolve()),
                )
            )

        preprocessed_subjects.append(
            PreprocessedSubject(
                subject_id=subject_id,
                variants=variants,
            )
        )

    # Publish the manifest once, after every subject is on disk.
    now = manifests.timestamp()
    preprocessed_manifest = PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        subjects=preprocessed_subjects,
        augmentation_factor=config.augmentation_factor,
    )
    preprocessed_manifest.write(layout.preprocessed_manifest_path())
