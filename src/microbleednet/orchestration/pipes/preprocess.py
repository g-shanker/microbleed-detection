import nibabel as nib
import numpy as np

from ...core import io
from ...core.datamodels import Modality
from ...core.engines import processor
from ...core.transforms import augmentations, frst
from ..configs import PreprocessConfig
from ..layouts import DatasetLayout
from ..manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    PreprocessedVariant,
    RawDatasetManifest,
)
from ..utils import resolve_path_string


def execute(config: PreprocessConfig) -> None:
    layout = DatasetLayout(dataset_dir=config.dataset_dir)
    raw_manifest = RawDatasetManifest.read(layout.raw_manifest_path())

    source_modalities: dict[str, Modality] = {
        source.source_id: source.modality for source in raw_manifest.sources
    }
    preprocessed_subjects: list[PreprocessedSubject] = []

    for subject in raw_manifest.subjects:
        subject_id = subject.subject_id

        raw_volume = io.load_volume(subject.volume_path)
        raw_mask = io.load_volume(subject.mask_path)

        modality = source_modalities[subject.source_id]
        preprocess_result = processor.preprocess(raw_volume, raw_mask, modality)

        variants: list[PreprocessedVariant] = []
        original_volume = preprocess_result.image
        original_mask = preprocess_result.mask
        for variant_index in range(config.augmentation_factor):
            variant_input_volume = original_volume
            variant_input_mask = original_mask
            if variant_index > 0:
                variant_input_volume, variant_input_mask = augmentations.augment(
                    variant_input_volume, variant_input_mask
                )

            variant_volume = nib.Nifti1Image(
                variant_input_volume, preprocess_result.affine
            )
            variant_mask = nib.Nifti1Image(
                variant_input_mask, preprocess_result.affine
            )
            variant_frst = frst.apply(np.asarray(variant_input_volume))
            variant_frst = nib.Nifti1Image(variant_frst, preprocess_result.affine)

            volume_path = layout.variant_volume_path(subject_id, variant_index)
            mask_path = layout.variant_mask_path(subject_id, variant_index)
            frst_path = layout.variant_frst_path(subject_id, variant_index)
            
            io.save_volume(variant_volume, volume_path)
            io.save_volume(variant_mask, mask_path)
            io.save_volume(variant_frst, frst_path)
            
            variants.append(
                PreprocessedVariant(
                    volume_path=resolve_path_string(volume_path),
                    mask_path=resolve_path_string(mask_path),
                    frst_path=resolve_path_string(frst_path),
                )
            )

        preprocessed_subjects.append(
            PreprocessedSubject(
                subject_id=subject_id,
                variants=variants,
            )
        )

    # Publish the manifest once, after every subject is on disk.
    preprocessed_manifest = PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
        subjects=preprocessed_subjects,
        augmentation_factor=config.augmentation_factor,
    )
    preprocessed_manifest.write(layout.preprocessed_manifest_path())
