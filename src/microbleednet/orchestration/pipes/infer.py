import logging
from pathlib import Path

import torch

from ...constants import (
    DETECTOR_THRESHOLD,
    DISCRIMINATOR_PATCH_SIZE,
    MAXIMUM_ELLIPTICITY,
    MINIMUM_BRAIN_DISTANCE_MM,
    MINIMUM_VOLUME_MM3,
    PREPROCESSING_AUGMENTATION_FACTOR,
    STUDENT_THRESHOLD,
    VARIANT_INDEX,
)
from ...core import io as core_io
from ...core.common.models import CandidateDetector, CandidateDiscriminatorStudent
from ...core.datamodels import Modality, VoxelSpacing
from ...core.engines import inference as core_inference
from ...core.engines import processor as core_processor
from ...errors import ApplicationError
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
from ..utils import (
    load_stage_checkpoint,
    release_gpu_memory,
    resolve_path_string,
)
from .preprocess import preprocess_subject

logger = logging.getLogger(__name__)


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
    logger.info(
        "Starting inference for %d subjects on %s; detector=%s student=%s",
        len(manifest.subjects),
        config.device,
        config.detector_checkpoint_path,
        config.student_checkpoint_path,
    )
    detector, student = load_inference_models(
        config.device,
        config.detector_checkpoint_path,
        config.student_checkpoint_path,
    )
    try:
        for subject in progress.track(
            manifest.subjects, description="Preprocessing inference subjects"
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
            detector,
            student,
        )
    finally:
        del detector
        del student
        release_gpu_memory()
    logger.info("Inference complete for %d subjects", len(subjects))


def load_inference_models(
    device_name: str,
    detector_checkpoint: Path,
    student_checkpoint: Path,
) -> tuple[CandidateDetector, CandidateDiscriminatorStudent]:
    """Load detector and student checkpoints onto the requested device."""
    device = torch.device(device_name)
    logger.info("Loading inference models on %s", device_name)
    detector = CandidateDetector().to(device)
    student = CandidateDiscriminatorStudent().to(device)
    try:
        load_stage_checkpoint(detector, detector_checkpoint, "detector")
        load_stage_checkpoint(student, student_checkpoint, "student")
    except Exception:
        del detector
        del student
        release_gpu_memory()
        raise
    return detector, student


def infer_subjects(
    device_name: str,
    output_layout: ExperimentLayout,
    detector_checkpoint: Path,
    student_checkpoint: Path,
    subjects: list[PreprocessedSubject],
    detector: CandidateDetector,
    student: CandidateDiscriminatorStudent,
) -> None:
    """Infer all preprocessed subjects and persist their output manifest."""
    results = [
        infer_subject(subject, detector, student, output_layout)
        for subject in progress.track(subjects, description="Inferring subjects")
    ]

    manifest = InferManifest(
        status=ManifestStatus.COMPLETE,
        device=device_name,
        detector_checkpoint_path=resolve_path_string(detector_checkpoint),
        student_checkpoint_path=resolve_path_string(student_checkpoint),
        detector_threshold=DETECTOR_THRESHOLD,
        student_threshold=STUDENT_THRESHOLD,
        discriminator_patch_size=DISCRIMINATOR_PATCH_SIZE,
        minimum_volume_mm3=MINIMUM_VOLUME_MM3,
        maximum_ellipticity=MAXIMUM_ELLIPTICITY,
        minimum_brain_distance_mm=MINIMUM_BRAIN_DISTANCE_MM,
        subjects=results,
    )
    try:
        manifest.write(output_layout.inference_manifest_path())
    except (OSError, ValueError) as error:
        raise ApplicationError(
            category="Output",
            summary="Could not write the inference manifest",
            cause=str(error),
            fix="Check the inference output directory and filesystem permissions",
            context={"path": str(output_layout.inference_manifest_path())},
        ) from error
    logger.info(
        "Inference manifest written to %s",
        output_layout.inference_manifest_path(),
    )


def infer_subject(
    subject: PreprocessedSubject,
    detector: CandidateDetector,
    student: CandidateDiscriminatorStudent,
    output_layout: ExperimentLayout,
) -> InferredSubject:
    """Produce and save a restored binary detection mask for one subject."""
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
        DISCRIMINATOR_PATCH_SIZE,
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
    try:
        core_io.save_volume(output_image, output_path)
    except OSError as error:
        raise ApplicationError(
            category="Output",
            summary="Could not write the inference output",
            cause=str(error),
            fix="Check the inference output directory and filesystem permissions",
            context={"path": str(output_path), "subject_id": subject.subject_id},
        ) from error

    return InferredSubject(
        subject_id=subject.subject_id,
        output_path=resolve_path_string(output_path),
    )


