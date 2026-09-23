"""Run the internal detector/discriminator inference pipeline."""

import logging

import torch

from ...core import io as core_io
from ...core.common.models import CandidateDetector, CandidateDiscriminatorStudent
from ...core.datamodels import Modality, PatchSizes, VoxelSpacing
from ...core.engines import inference as core_inference
from ...core.engines import processor as core_processor
from ...progress import progress
from ..configs import InferConfig
from ..layouts import DatasetLayout, ExperimentLayout
from ..manifests import (
    InferManifest,
    InferredSubject,
    ManifestStatus,
    PreprocessedSubject,
    RawDatasetManifest,
)
from ..utils import release_gpu_memory, resolve_path_string
from .preprocess import preprocess_subject

logger = logging.getLogger(__name__)

VARIANT_INDEX = 0
DETECTOR_THRESHOLD = 0.5
STUDENT_THRESHOLD = 0.5
MINIMUM_VOLUME_MM3 = 2.5
MAXIMUM_ELLIPTICITY = 0.2
MINIMUM_BRAIN_DISTANCE_MM = 5.0
PREPROCESSING_AUGMENTATION_FACTOR = 1


def execute(config: InferConfig) -> None:
    """Preprocess raw subjects and infer final binary detection masks."""
    layout = ExperimentLayout(experiment_dir=config.output_dir)
    manifest = RawDatasetManifest.read(
        DatasetLayout(dataset_dir=config.dataset_dir).raw_manifest_path()
    )
    modalities: dict[str, Modality] = {
        source.source_id: source.modality for source in manifest.sources
    }
    preprocessing_layout = DatasetLayout(dataset_dir=config.output_dir)
    subjects: list[PreprocessedSubject] = []
    for subject in progress.track(
        manifest.subjects, "Preprocessing inference subjects"
    ):
        subjects.append(
            preprocess_subject(
                subject,
                modalities[subject.source_id],
                preprocessing_layout,
                PREPROCESSING_AUGMENTATION_FACTOR,
            )
        )
    infer_subjects(
        config.device,
        layout,
        config.detector_checkpoint_path,
        config.student_checkpoint_path,
        subjects,
    )


def infer_subjects(
    device_name: str,
    output_layout: ExperimentLayout,
    detector_checkpoint,
    student_checkpoint,
    subjects: list[PreprocessedSubject],
) -> None:
    logger.info("Running inference for %d subjects on %s.", len(subjects), device_name)
    device = torch.device(device_name)
    detector = CandidateDetector().to(device)
    student = CandidateDiscriminatorStudent().to(device)
    try:
        core_io.load_model_weights(detector, detector_checkpoint)

        core_io.load_model_weights(student, student_checkpoint)

        results = [
            infer_subject(subject, detector, student, output_layout)
            for subject in progress.track(subjects, "Inferring subjects")
        ]

        InferManifest(
            status=ManifestStatus.COMPLETE,
            device=device_name,
            detector_checkpoint_path=resolve_path_string(detector_checkpoint),
            student_checkpoint_path=resolve_path_string(student_checkpoint),
            detector_threshold=DETECTOR_THRESHOLD,
            student_threshold=STUDENT_THRESHOLD,
            discriminator_patch_size=PatchSizes.DISCRIMINATOR,
            minimum_volume_mm3=MINIMUM_VOLUME_MM3,
            maximum_ellipticity=MAXIMUM_ELLIPTICITY,
            minimum_brain_distance_mm=MINIMUM_BRAIN_DISTANCE_MM,
            subjects=results,
        ).write(output_layout.inference_manifest_path())
        
        logger.info(
            "Inference complete for %d subjects; manifest written to %s.",
            len(results),
            output_layout.inference_manifest_path(),
        )
    finally:
        del detector
        del student
        release_gpu_memory()


def infer_subject(
    subject: PreprocessedSubject,
    detector: CandidateDetector,
    student: CandidateDiscriminatorStudent,
    output_layout: ExperimentLayout,
) -> InferredSubject:
    volume_image = core_io.load_volume(subject.variants[VARIANT_INDEX].volume_path)
    volume = core_io.nifti_to_numpy(volume_image)
    frst_array = core_io.nifti_to_numpy(
        core_io.load_volume(subject.variants[VARIANT_INDEX].frst_path)
    )

    detector_probability = core_inference.infer_detector(detector, volume, frst_array)
    retained_labels = core_inference.infer_discriminator(
        student,
        volume,
        frst_array,
        detector_probability,
        DETECTOR_THRESHOLD,
        PatchSizes.DISCRIMINATOR,
        STUDENT_THRESHOLD,
    )

    zooms = list(volume_image.header.get_zooms())
    voxel_sizes: VoxelSpacing = (
        float(zooms[0]),
        float(zooms[1]),
        float(zooms[2]),
    )
    final_mask_array = core_processor.postprocess(
        retained_labels,
        volume,
        voxel_sizes,
        MINIMUM_VOLUME_MM3,
        MAXIMUM_ELLIPTICITY,
        MINIMUM_BRAIN_DISTANCE_MM,
    )

    output_image = core_processor.volume_ops.restore_cropped_volume(
        final_mask_array,
        subject.bounding_box,
        core_io.load_volume(subject.original_volume_path),
    )
    output_path = output_layout.inference_output_path(subject.subject_id)
    core_io.save_volume(output_image, output_path)

    return InferredSubject(
        subject_id=subject.subject_id,
        output_path=resolve_path_string(output_path),
    )


