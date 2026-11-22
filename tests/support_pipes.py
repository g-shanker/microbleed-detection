"""Builders shared by the pipe tests (train, patch, infer, evaluate)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from microbleednet.core.common.models import CandidateDetector
from microbleednet.orchestration.layouts import ExperimentLayout
from microbleednet.orchestration.manifests import (
    InferManifest,
    InferredSubject,
    ManifestStatus,
    PreprocessedSubject,
    SplitManifest,
)
from tests.support import (
    preprocessed_subjects,
    save_nifti,
    write_preprocessed_manifest,
    write_raw_manifest,
    write_split_manifest,
)


class StubDetector(CandidateDetector):
    """A ``CandidateDetector`` without weights, for isinstance-validated configs."""

    def __init__(self) -> None:
        super(CandidateDetector, self).__init__()


@dataclass(frozen=True)
class ExperimentInputs:
    root: Path
    subjects: list[PreprocessedSubject]
    split: SplitManifest


def write_experiment_inputs(
    tmp_path: Path, count: int = 3, variants: int = 1, **nifti_kwargs: Any
) -> ExperimentInputs:
    """Write a complete preprocessed dataset and split under ``tmp_path``.

    The last subject is the test subject and the one before it is validation;
    the rest train. ``write_training_dataset`` cannot be used here because it
    never sets the manifest's ``augmentation_factor`` for multi-variant subjects.
    """
    subjects = preprocessed_subjects(
        count, tmp_path / "inputs", variants=variants, **nifti_kwargs
    )
    write_preprocessed_manifest(
        tmp_path / "dataset", subjects, augmentation_factor=variants
    )
    ids = [subject.subject_id for subject in subjects]
    split = write_split_manifest(
        tmp_path, ids[:-2], validation_ids=ids[-2:-1], test_ids=ids[-1:]
    )
    return ExperimentInputs(tmp_path, subjects, split)


def write_explicit_inputs(tmp_path: Path, **raw_kwargs: Any) -> dict[str, Any]:
    """Write a raw dataset and checkpoints; return Infer/Evaluate config values."""
    dataset_dir = tmp_path / "dataset"
    write_raw_manifest(dataset_dir, **raw_kwargs)
    checkpoints = {}
    for name in ("detector", "student"):
        checkpoints[f"{name}_checkpoint_path"] = tmp_path / f"{name}.pth"
        checkpoints[f"{name}_checkpoint_path"].touch()
    return {
        "dataset_dir": dataset_dir,
        "output_dir": tmp_path / "output",
        "bias_field_correction": "fast",
        "use_amp": False,
        "device": "cpu",
        **checkpoints,
    }


def write_inference_manifest(
    layout: ExperimentLayout,
    directory: Path,
    predictions: dict[str, np.ndarray],
    status: ManifestStatus = ManifestStatus.COMPLETE,
) -> InferManifest:
    """Write real prediction NIfTIs and the inference manifest that lists them."""
    subjects = [
        InferredSubject(
            subject_id=subject_id,
            detector_output_path=str(directory / f"{subject_id}-detector.nii.gz"),
            discriminator_output_path=save_nifti(
                directory / f"{subject_id}-prediction.nii.gz", data=prediction
            ),
        )
        for subject_id, prediction in predictions.items()
    ]
    manifest = InferManifest(
        status=status,
        device="cpu",
        detector_checkpoint_path="detector.pth",
        student_checkpoint_path="student.pth",
        detector_threshold=0.5,
        student_threshold=0.5,
        discriminator_patch_size=24,
        minimum_volume_mm3=2.5,
        maximum_ellipticity=0.2,
        minimum_brain_distance_mm=5.0,
        subjects=subjects,
    )
    manifest.write(layout.inference_manifest_path())
    return manifest
