"""Run the internal detector/discriminator inference pipeline."""

from typing import cast

import torch

from ...core import io as core_io
from ...core.common.models import CandidateDetector, CandidateDiscriminatorStudent
from ...core.datamodels import PatchSizes
from ...core.engines import inference as core_inference
from ...core.engines import processor as core_processor
from ..configs import InferConfig
from ..layouts import ExperimentLayout
from ..manifests import (
    InferManifest,
    InferredSubject,
    ManifestStatus,
)
from ..utils import release_gpu_memory

DETECTOR_STAGE = "detector"
STUDENT_STAGE = "student"
VARIANT_INDEX = 0
DETECTOR_THRESHOLD = 0.5
STUDENT_THRESHOLD = 0.5
MINIMUM_VOLUME_MM3 = 2.5
MAXIMUM_ELLIPTICITY = 0.2
MINIMUM_BRAIN_DISTANCE_MM = 5.0


def execute(config: InferConfig) -> None:
    """Infer final binary detection masks for configured preprocessed subjects."""
    layout = ExperimentLayout(experiment_dir=config.experiment_dir)
    device = torch.device(config.device)
    detector = CandidateDetector().to(device)
    student = CandidateDiscriminatorStudent().to(device)
    try:
        detector_checkpoint = layout.best_checkpoint_path(DETECTOR_STAGE)
        core_io.load_model_weights(detector, detector_checkpoint)

        student_checkpoint = layout.best_checkpoint_path(STUDENT_STAGE)
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
            
            final_mask_array = core_processor.postprocess(
                retained_labels,
                volume,
                cast(tuple[float, float, float], volume_image.header.get_zooms()[:3]),
                MINIMUM_VOLUME_MM3,
                MAXIMUM_ELLIPTICITY,
                MINIMUM_BRAIN_DISTANCE_MM,
            )

            final_mask_image = core_io.numpy_to_nifti(final_mask_array, volume_image)

            output_path = layout.inference_output_path(subject.subject_id)
            core_io.save_volume(final_mask_image, output_path)

            results.append(
                InferredSubject(
                    subject_id=subject.subject_id,
                    output_path=str(output_path.resolve()),
                )
            )

        InferManifest(
            status=ManifestStatus.COMPLETE,
            device=config.device,
            detector_checkpoint_path=str(detector_checkpoint.resolve()),
            student_checkpoint_path=str(student_checkpoint.resolve()),
            detector_threshold=DETECTOR_THRESHOLD,
            student_threshold=STUDENT_THRESHOLD,
            discriminator_patch_size=PatchSizes.DISCRIMINATOR,
            minimum_volume_mm3=MINIMUM_VOLUME_MM3,
            maximum_ellipticity=MAXIMUM_ELLIPTICITY,
            minimum_brain_distance_mm=MINIMUM_BRAIN_DISTANCE_MM,
            subjects=results,
        ).write(layout.inference_manifest_path())
    finally:
        del detector
        del student
        release_gpu_memory()


