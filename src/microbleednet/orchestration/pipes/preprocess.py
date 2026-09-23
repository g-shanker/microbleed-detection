import logging

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
    RawSubject,
    content_fingerprint,
)
from ..utils import resolve_path_string

logger = logging.getLogger(__name__)


def execute(config: PreprocessConfig) -> None:
    layout = DatasetLayout(dataset_dir=config.dataset_dir)
    raw_manifest = RawDatasetManifest.read(layout.raw_manifest_path())
    preprocessed_manifest_path = layout.preprocessed_manifest_path()
    raw_manifest_fingerprint = content_fingerprint(raw_manifest)

    source_modalities: dict[str, Modality] = {
        source.source_id: source.modality for source in raw_manifest.sources
    }
    preprocessed_subjects: list[PreprocessedSubject] = []
    skipped_subjects = 0
    if config.resume:
        existing_manifest = PreprocessedDatasetManifest.read(
            preprocessed_manifest_path
        )
        preprocessed_subjects.extend(existing_manifest.subjects)

    logger.info(
        "Preprocessing %d subjects with augmentation factor %d (resume=%s).",
        len(raw_manifest.subjects),
        config.augmentation_factor,
        config.resume,
    )

    PreprocessedDatasetManifest(
        status=ManifestStatus.RUNNING,
        subjects=preprocessed_subjects,
        augmentation_factor=config.augmentation_factor,
        raw_manifest_fingerprint=raw_manifest_fingerprint,
    ).write(preprocessed_manifest_path)

    for subject in progress.track(raw_manifest.subjects, "Preprocessing subjects"):
        subject_id = subject.subject_id
        if any(
            completed.subject_id == subject_id
            for completed in preprocessed_subjects
        ):
            skipped_subjects += 1
            continue

        preprocessed_subject = preprocess_subject(
            subject,
            source_modalities[subject.source_id],
            layout,
            config.augmentation_factor,
        )
        preprocessed_subjects.append(preprocessed_subject)
        PreprocessedDatasetManifest(
            status=ManifestStatus.RUNNING,
            subjects=preprocessed_subjects,
            augmentation_factor=config.augmentation_factor,
            raw_manifest_fingerprint=raw_manifest_fingerprint,
        ).write(preprocessed_manifest_path)

    PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
        subjects=preprocessed_subjects,
        augmentation_factor=config.augmentation_factor,
        raw_manifest_fingerprint=raw_manifest_fingerprint,
    ).write(preprocessed_manifest_path)
    logger.info(
        "Preprocessed %d subjects (%d skipped); manifest written to %s.",
        len(preprocessed_subjects),
        skipped_subjects,
        preprocessed_manifest_path,
    )


def preprocess_subject(
    subject: RawSubject,
    modality: Modality,
    layout: DatasetLayout,
    augmentation_factor: int,
) -> PreprocessedSubject:
    raw_volume = io.load_volume(subject.volume_path)
    raw_mask = io.load_volume(subject.mask_path) if subject.mask_path else None
    preprocess_output = processor.preprocess(
        PreprocessInput(raw_volume, raw_mask, modality)
    )
    processed_volume = preprocess_output.volume
    processed_mask = preprocess_output.mask
    processed_affine = preprocess_output.affine

    variants: list[PreprocessedVariant] = []
    for variant_index in range(augmentation_factor):
        variant_input_volume = processed_volume
        variant_input_mask = processed_mask
        if variant_index > 0:
            variant_input_volume, variant_input_mask = augmentations.augment(
                variant_input_volume, variant_input_mask
            )

        variant_volume = nib.Nifti1Image(variant_input_volume, processed_affine)
        variant_frst = nib.Nifti1Image(
            frst.apply(np.asarray(variant_input_volume)), processed_affine
        )
        volume_path = layout.variant_volume_path(subject.subject_id, variant_index)
        frst_path = layout.variant_frst_path(subject.subject_id, variant_index)
        io.save_volume(variant_volume, volume_path)
        io.save_volume(variant_frst, frst_path)

        mask_path = None
        if variant_input_mask is not None:
            variant_mask = nib.Nifti1Image(variant_input_mask, processed_affine)
            mask_path = layout.variant_mask_path(subject.subject_id, variant_index)
            io.save_volume(variant_mask, mask_path)

        variants.append(
            PreprocessedVariant(
                volume_path=resolve_path_string(volume_path),
                frst_path=resolve_path_string(frst_path),
                mask_path=(
                    resolve_path_string(mask_path) if mask_path is not None else None
                ),
            )
        )

    return PreprocessedSubject(
        subject_id=subject.subject_id,
        original_volume_path=subject.volume_path,
        bounding_box=preprocess_output.bounding_box,
        variants=variants,
    )
