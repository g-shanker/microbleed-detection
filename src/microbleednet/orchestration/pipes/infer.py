"""Run the internal detector/discriminator inference pipeline."""

from dataclasses import dataclass
from typing import Callable

import nibabel as nib
import torch

from ...core import io as core_io
from ...core.common.models import CandidateDetector, CandidateDiscriminatorStudent
from ...core.datamodels import Modality, PatchSizes, PreprocessInput, VoxelSpacing
from ...core.engines import inference as core_inference
from ...core.engines import processor as core_processor
from ...core.transforms import frst
from ..configs import InferConfig, InferExperimentConfig, InferExplicitConfig
from ..layouts import DETECTOR_STAGE, STUDENT_STAGE, DatasetLayout, ExperimentLayout
from ..manifests import (
    InferManifest,
    InferredSubject,
    ManifestStatus,
    RawDatasetManifest,
    RawSubject,
)
from ..utils import release_gpu_memory, resolve_path_string

VARIANT_INDEX = 0
DETECTOR_THRESHOLD = 0.5
STUDENT_THRESHOLD = 0.5
MINIMUM_VOLUME_MM3 = 2.5
MAXIMUM_ELLIPTICITY = 0.2
MINIMUM_BRAIN_DISTANCE_MM = 5.0


@dataclass(frozen=True)
class InferenceInput:
    subject_id: str
    volume: nib.Nifti1Image
    frst: nib.Nifti1Image
    restore: Callable[[object], nib.Nifti1Image] | None = None


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
    infer_preprocessed_subjects(
        config,
        layout,
        layout.best_checkpoint_path(DETECTOR_STAGE),
        layout.best_checkpoint_path(STUDENT_STAGE),
        mode.subjects,
    )


def execute_explicit(config: InferConfig, mode: InferExplicitConfig) -> None:
    """Infer using explicitly supplied checkpoints and output directory."""
    layout = ExperimentLayout(experiment_dir=mode.output_dir)
    manifest = RawDatasetManifest.read(
        DatasetLayout(dataset_dir=mode.dataset_dir).raw_manifest_path()
    )
    modalities = {
        source.source_id: source.modality for source in manifest.sources
    }
    infer_raw_subjects(
        config.device,
        layout,
        mode.detector_checkpoint_path,
        mode.student_checkpoint_path,
        manifest.subjects,
        modalities,
    )


def infer_preprocessed_subjects(
    config: InferConfig,
    output_layout: ExperimentLayout,
    detector_checkpoint,
    student_checkpoint,
    subjects,
) -> None:
    records = [
        InferenceInput(
            subject.subject_id,
            core_io.load_volume(subject.variants[VARIANT_INDEX].volume_path),
            core_io.load_volume(subject.variants[VARIANT_INDEX].frst_path),
        )
        for subject in subjects
    ]
    execute_records(
        config.device, output_layout, detector_checkpoint, student_checkpoint, records
    )


def execute_records(
    device_name: str,
    output_layout: ExperimentLayout,
    detector_checkpoint,
    student_checkpoint,
    records: list[InferenceInput],
) -> None:
    device = torch.device(device_name)
    detector = CandidateDetector().to(device)
    student = CandidateDiscriminatorStudent().to(device)
    try:
        core_io.load_model_weights(detector, detector_checkpoint)

        core_io.load_model_weights(student, student_checkpoint)

        results: list[InferredSubject] = []
        for record in records:
            volume_image = record.volume
            volume = core_io.nifti_to_numpy(volume_image)
            frst_array = core_io.nifti_to_numpy(record.frst)

            detector_probability = core_inference.infer_detector(
                detector, volume, frst_array
            )
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

            final_mask_image = core_io.numpy_to_nifti(final_mask_array, volume_image)

            output_image = (
                record.restore(final_mask_array)
                if record.restore is not None
                else final_mask_image
            )
            output_path = output_layout.inference_output_path(record.subject_id)
            core_io.save_volume(output_image, output_path)

            results.append(
                InferredSubject(
                    subject_id=record.subject_id,
                    output_path=resolve_path_string(output_path),
                )
            )

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
    finally:
        del detector
        del student
        release_gpu_memory()


def infer_raw_subjects(
    device_name: str,
    output_layout: ExperimentLayout,
    detector_checkpoint,
    student_checkpoint,
    subjects: list[RawSubject],
    modalities: dict[str, Modality],
) -> None:
    """Preprocess raw subjects and run inference with explicit checkpoints."""
    records: list[InferenceInput] = []
    for subject in subjects:
        raw_volume = core_io.load_volume(subject.volume_path)
        raw_mask = (
            core_io.load_volume(subject.mask_path)
            if subject.mask_path is not None
            else None
        )
        preprocess_result = core_processor.preprocess(
            PreprocessInput(raw_volume, raw_mask, modalities[subject.source_id])
        )
        processed_volume = nib.Nifti1Image(
            preprocess_result.volume, preprocess_result.affine
        )
        processed_frst = nib.Nifti1Image(
            frst.apply(preprocess_result.volume), preprocess_result.affine
        )
        records.append(
            InferenceInput(
                subject.subject_id,
                processed_volume,
                processed_frst,
                lambda prediction, result=preprocess_result: (
                    core_processor.volume_ops.restore_cropped_volume(
                        prediction,
                        result.bounding_box,
                        result.original_volume,
                    )
                ),
            )
        )
    execute_records(
        device_name,
        output_layout,
        detector_checkpoint,
        student_checkpoint,
        records,
    )


