"""Run the internal detector/discriminator inference pipeline."""

import torch

from ...core import io as core_io
from ...core.common.models import CandidateDetector, CandidateDiscriminatorStudent
from ...core.datamodels import PatchSizes, VoxelSpacing
from ...core.engines import inference as core_inference
from ...core.engines import processor as core_processor
from ..configs import InferConfig, InferExperimentConfig, InferExplicitConfig
from ..layouts import DETECTOR_STAGE, STUDENT_STAGE, ExperimentLayout
from ..manifests import (
    InferManifest,
    InferredSubject,
    ManifestStatus,
)
from ..utils import release_gpu_memory, resolve_path_string

VARIANT_INDEX = 0
DETECTOR_THRESHOLD = 0.5
STUDENT_THRESHOLD = 0.5
MINIMUM_VOLUME_MM3 = 2.5
MAXIMUM_ELLIPTICITY = 0.2
MINIMUM_BRAIN_DISTANCE_MM = 5.0


def execute(config: InferConfig) -> None:
    """Infer final binary detection masks for configured preprocessed subjects."""
    if config.experiment is not None:
        execute_experiment(config, config.experiment)
    else:
        assert config.explicit is not None
        execute_explicit(config, config.explicit)


def execute_experiment(
    config: InferConfig, mode: InferExperimentConfig
) -> None:
    """Infer using checkpoints resolved from an experiment directory."""
    layout = ExperimentLayout(experiment_dir=mode.experiment_dir)
    execute_inference(
        config,
        layout,
        layout.best_checkpoint_path(DETECTOR_STAGE),
        layout.best_checkpoint_path(STUDENT_STAGE),
    )


def execute_explicit(config: InferConfig, mode: InferExplicitConfig) -> None:
    """Infer using explicitly supplied checkpoints and output directory."""
    layout = ExperimentLayout(experiment_dir=mode.output_dir)
    execute_inference(
        config,
        layout,
        mode.detector_checkpoint_path,
        mode.student_checkpoint_path,
    )


def execute_inference(
    config: InferConfig,
    output_layout: ExperimentLayout,
    detector_checkpoint,
    student_checkpoint,
) -> None:
    device = torch.device(config.device)
    detector = CandidateDetector().to(device)
    student = CandidateDiscriminatorStudent().to(device)
    try:
        core_io.load_model_weights(detector, detector_checkpoint)

        core_io.load_model_weights(student, student_checkpoint)

        results: list[InferredSubject] = []
        for subject in config.subjects:
            variant = subject.variants[VARIANT_INDEX]
            volume_image = core_io.load_volume(variant.volume_path)
            volume = core_io.nifti_to_numpy(volume_image)
            frst = core_io.nifti_to_numpy(core_io.load_volume(variant.frst_path))

            detector_probability = core_inference.infer_detector(
                detector, volume, frst
            )
            retained_labels = core_inference.infer_discriminator(
                student,
                volume,
                frst,
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

            final_mask_image = core_io.numpy_to_nifti(final_mask_array, volume_image)

            output_path = output_layout.inference_output_path(subject.subject_id)
            core_io.save_volume(final_mask_image, output_path)

            results.append(
                InferredSubject(
                    subject_id=subject.subject_id,
                    output_path=resolve_path_string(output_path),
                )
            )

        InferManifest(
            status=ManifestStatus.COMPLETE,
            device=config.device,
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
    finally:
        del detector
        del student
        release_gpu_memory()


