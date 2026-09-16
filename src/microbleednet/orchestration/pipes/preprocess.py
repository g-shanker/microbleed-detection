import nibabel as nib
import numpy as np

from ...core import io
from ...core.datamodels import Modality, PreprocessInput
from ...core.engines import processor
from ...core.transforms import augmentations, frst
from ...progress import progress
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

    for subject in progress.track(raw_manifest.subjects, "Preprocessing subjects"):
        subject_id = subject.subject_id

        raw_volume = io.load_volume(subject.volume_path)
        raw_mask = io.load_volume(subject.mask_path) if subject.mask_path else None

        modality = source_modalities[subject.source_id]
        processed_volume, processed_mask, processed_affine = processor.preprocess(
            PreprocessInput(raw_volume, raw_mask, modality)
        )

        variants: list[PreprocessedVariant] = []
        original_volume = processed_volume
        original_mask = processed_mask
        for variant_index in range(config.augmentation_factor):
            variant_input_volume = original_volume
            variant_input_mask = original_mask
            if variant_index > 0:
                variant_input_volume, variant_input_mask = augmentations.augment(
                    variant_input_volume, variant_input_mask
                )

            variant_volume = nib.Nifti1Image(
                variant_input_volume, processed_affine
            )
            variant_frst = frst.apply(np.asarray(variant_input_volume))
            variant_frst = nib.Nifti1Image(variant_frst, processed_affine)

            volume_path = layout.variant_volume_path(subject_id, variant_index)
            frst_path = layout.variant_frst_path(subject_id, variant_index)

            io.save_volume(variant_volume, volume_path)
            io.save_volume(variant_frst, frst_path)

            mask_path = None
            if variant_input_mask is not None:
                variant_mask = nib.Nifti1Image(
                    variant_input_mask, processed_affine
                )
                mask_path = layout.variant_mask_path(subject_id, variant_index)
                io.save_volume(variant_mask, mask_path)

            variants.append(
                PreprocessedVariant(
                    volume_path=resolve_path_string(volume_path),
                    frst_path=resolve_path_string(frst_path),
                    mask_path=(
                        resolve_path_string(mask_path)
                        if mask_path is not None
                        else None
                    ),
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
