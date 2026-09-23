"""Top-level command configs the orchestration layer consumes.

Each pipe's ``execute()`` takes one of these config objects whole. They live
here because the orchestration layer owns them; the CLI imports them to parse
TOML into already-validated objects, which keeps the dependency direction
``cli -> orchestration`` intact. Loading a config from a TOML file is a CLI
concern (``cli/utils.py``).
"""

import math
import re
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from ..core.datamodels import FrozenModel, Hyperparameters, Modality
from .layouts import (
    DETECTOR_STAGE,
    STUDENT_STAGE,
    TEACHER_STAGE,
    DatasetLayout,
    ExperimentLayout,
    SplitName,
    StageName,
)
from .manifests import (
    Manifest,
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    RawDatasetManifest,
    RawSubject,
    SplitManifest,
    TrainManifest,
    TrainStageManifest,
    content_fingerprint,
)

# Token a volume/mask filename pattern must contain at least once; the text it
# matches becomes the subject ID. Shared by the index-data pipeline (which
# splits filenames on it) and IndexDataConfig (which validates its presence).
SUBJECT_ID_PLACEHOLDER = "{subject_id}"
SOURCE_ID_PLACEHOLDER = "{source_id}"

# A source_id namespaces subject IDs and becomes part of on-disk paths, so it is
# restricted to filesystem-safe characters.
SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def ensure_device_available(device_name: str) -> None:
    """Validate a device string without importing torch at module load time."""
    import torch

    try:
        device = torch.device(device_name)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid device: {device_name}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device is not available")


def ensure_directory_exists(directory: Path, description: str) -> None:
    if not directory.is_dir():
        raise ValueError(f"{description} does not exist: {directory}")


def ensure_file_exists(file_path: Path, description: str) -> None:
    if not file_path.is_file():
        raise ValueError(f"{description} does not exist: {file_path}")


def ensure_manifest_complete(
    manifest: Manifest, manifest_path: Path, description: str
) -> None:
    if manifest.status is not ManifestStatus.COMPLETE:
        raise ValueError(
            f"{description} at {manifest_path} has status "
            f"{manifest.status.value!r}; a consumer may only read a "
            "complete manifest."
        )


def ensure_manifest_not_complete(manifest: Manifest, error_message: str) -> None:
    if manifest.status is ManifestStatus.COMPLETE:
        raise ValueError(error_message)


def ensure_training_resume_state(experiment_layout: ExperimentLayout) -> None:
    for stage in (DETECTOR_STAGE, TEACHER_STAGE, STUDENT_STAGE):
        if experiment_layout.latest_checkpoint_path(stage).is_file():
            return

        stage_manifest_path = experiment_layout.stage_manifest_path(stage)
        if (
            stage_manifest_path.is_file()
            and TrainStageManifest.read(stage_manifest_path).status
            is ManifestStatus.COMPLETE
        ):
            return

    raise ValueError(
        "cannot resume: no training stage checkpoint or completion manifest exists"
    )


def ensure_subjects_have_masks(
    subjects: list[RawSubject] | list[PreprocessedSubject], description: str
) -> None:
    maskless_subjects = [
        subject.subject_id
        for subject in subjects
        if (
            subject.mask_path is None
            if isinstance(subject, RawSubject)
            else any(variant.mask_path is None for variant in subject.variants)
        )
    ]
    if maskless_subjects:
        raise ValueError(f"{description}: " + ", ".join(maskless_subjects))


def ensure_fingerprint_matches(
    actual_fingerprint: str,
    expected_manifest: Manifest,
    error_message: str,
) -> None:
    if actual_fingerprint != content_fingerprint(expected_manifest):
        raise ValueError(error_message)


class IndexDataConfig(FrozenModel):
    dataset_dir: Path = Field(
        description="Directory to which the indexed dataset manifests are written.",
    )
    input_dir: Path = Field(
        description="Input directory containing the volumes to index.",
    )
    volume_pattern: str = Field(
        description=(
            "Naming pattern of volumes in the input directory. "
            f"Must contain the '{SUBJECT_ID_PLACEHOLDER}' placeholder and "
            "the complete filename extension, such as '.nii.gz'."
        ),
    )
    mask_dir: Path | None = Field(
        default=None,
        description="Optional directory containing the complete mask volumes to index.",
    )
    mask_pattern: str | None = Field(
        default=None,
        description=(
            "Optional naming pattern of masks in the mask directory. "
            f"Must contain the '{SUBJECT_ID_PLACEHOLDER}' placeholder and "
            "the complete filename extension, such as '.nii.gz'."
        ),
    )
    source_id: str = Field(
        description=(
            "Namespace prepended to each subject ID as "
            f"'{SOURCE_ID_PLACEHOLDER}_{SUBJECT_ID_PLACEHOLDER}'. "
            "Allowed characters: letters, digits, '-', '_'."
        ),
    )
    modality: Modality = Field(
        description="Imaging modality of this source's volumes.",
    )

    @model_validator(mode="after")
    def validate_config(self) -> "IndexDataConfig":
        if not SOURCE_ID_PATTERN.match(self.source_id):
            raise ValueError(
                f"{SOURCE_ID_PLACEHOLDER} may contain only letters, digits, "
                "'-', and '_'"
            )

        patterns = [("volume_pattern", self.volume_pattern)]
        if self.mask_pattern is not None:
            patterns.append(("mask_pattern", self.mask_pattern))
        for name, pattern in patterns:
            if pattern.count(SUBJECT_ID_PLACEHOLDER) < 1:
                raise ValueError(
                    f"{name} must contain the '{SUBJECT_ID_PLACEHOLDER}' "
                    "placeholder at least once"
                )

        ensure_directory_exists(self.input_dir, "input_dir")
        if self.mask_dir is not None:
            ensure_directory_exists(self.mask_dir, "mask_dir")
        if (self.mask_dir is None) != (self.mask_pattern is None):
            raise ValueError("mask_dir and mask_pattern must be provided together")

        return self


class PreprocessConfig(FrozenModel):
    dataset_dir: Path = Field(
        description="Indexed dataset directory containing manifests/raw.json."
    )
    augmentation_factor: int = Field(
        gt=0,
        description=("Total persisted variants per subject, including the original."),
    )
    resume: bool = Field(
        default=False,
        description="Resume from the last per-subject preprocessing checkpoint.",
    )

    @model_validator(mode="after")
    def validate_config(self) -> "PreprocessConfig":
        ensure_directory_exists(self.dataset_dir, "dataset_dir")

        layout = DatasetLayout(dataset_dir=self.dataset_dir)
        manifest_path = layout.raw_manifest_path()
        ensure_file_exists(manifest_path, "raw manifest")

        raw_manifest = RawDatasetManifest.read(layout.raw_manifest_path())
        
        if not self.resume:
            return self

        preprocessed_manifest_path = layout.preprocessed_manifest_path()
        ensure_file_exists(
            preprocessed_manifest_path,
            "cannot resume: the preprocessed manifest",
        )

        existing_manifest = PreprocessedDatasetManifest.read(preprocessed_manifest_path)

        ensure_manifest_not_complete(
            existing_manifest,
            "cannot resume: the preprocessing run is already complete",
        )

        ensure_fingerprint_matches(
            existing_manifest.raw_manifest_fingerprint,
            raw_manifest,
            "cannot resume: the raw dataset manifest has changed",
        )

        return self


class SplitConfig(FrozenModel):
    dataset_dir: Path = Field(
        description="Preprocessed dataset directory containing subjects to split."
    )
    experiment_dir: Path = Field(
        description="Experiment directory to receive the split manifest."
    )
    train_size: float = Field(
        gt=0,
        lt=1,
        description="Proportion of subjects assigned to the training split.",
    )
    validation_size: float = Field(
        gt=0,
        lt=1,
        description="Proportion of subjects assigned to the validation split.",
    )
    test_size: float = Field(
        gt=0,
        lt=1,
        description="Proportion of subjects held out for testing.",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description="Optional random seed for reproducible subject splitting.",
    )

    @model_validator(mode="after")
    def validate_config(self) -> "SplitConfig":
        split_total = self.train_size + self.validation_size + self.test_size
        if not math.isclose(split_total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "train_size, validation_size, and test_size must sum to 1.0"
            )

        ensure_directory_exists(self.dataset_dir, "dataset_dir")

        layout = DatasetLayout(dataset_dir=self.dataset_dir)
        manifest_path = layout.preprocessed_manifest_path()
        ensure_file_exists(manifest_path, "preprocessed manifest")

        manifest = PreprocessedDatasetManifest.read(manifest_path)
        
        ensure_manifest_complete(
            manifest,
            manifest_path,
            "cannot split manifest",
        )

        ensure_subjects_have_masks(
            manifest.subjects,
            "cannot split a preprocessed dataset with subjects missing masks",
        )
        
        return self


class TrainConfig(FrozenModel):
    dataset_dir: Path = Field(
        description=(
            "Indexed dataset directory containing manifests/preprocessed.json."
        ),
    )
    experiment_dir: Path = Field(
        description=(
            "Directory to write per-stage patches, checkpoints, and manifests."
        ),
    )
    device: str = Field(
        description="Torch device string, e.g. 'cpu' or 'cuda'.",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description="Optional random seed for reproducible training runs.",
    )
    detector_candidate_threshold: float = Field(
        ge=0,
        le=1,
        description="Minimum detector probability retained as a student candidate.",
    )
    detector_augmentation_factor: int = Field(
        gt=0,
        description=(
            "Total variants used for detector training, including the original."
        ),
    )
    discriminator_augmentation_factor: int = Field(
        gt=0,
        description=(
            "Total variants used for discriminator training, including the original."
        ),
    )
    num_workers: int = Field(
        ge=0,
        description="Number of worker processes used by training data loaders.",
    )
    pin_memory: bool = Field(
        description=(
            "Whether training data loaders use page-locked host memory for transfers."
        ),
    )
    detector_hyperparameters: Hyperparameters = Field(
        description="Optimizer and training-loop settings for detector training.",
    )
    teacher_hyperparameters: Hyperparameters = Field(
        description="Optimizer and training-loop settings for teacher training.",
    )
    student_hyperparameters: Hyperparameters = Field(
        description="Optimizer and training-loop settings for student training.",
    )
    resume: bool = Field(
        default=False,
        description=(
            "Resume an interrupted training run from its latest stage checkpoint."
        ),
    )

    @model_validator(mode="after")
    def validate_config(self) -> "TrainConfig":
        ensure_device_available(self.device)

        ensure_directory_exists(self.dataset_dir, "dataset_dir")

        dataset_layout = DatasetLayout(dataset_dir=self.dataset_dir)
        preprocessed_manifest_path = dataset_layout.preprocessed_manifest_path()
        ensure_file_exists(preprocessed_manifest_path, "preprocessed manifest")

        preprocessed_manifest = PreprocessedDatasetManifest.read(
            preprocessed_manifest_path
        )

        ensure_manifest_complete(
            preprocessed_manifest,
            preprocessed_manifest_path,
            "manifest",
        )

        experiment_layout = ExperimentLayout(experiment_dir=self.experiment_dir)
        split_manifest_path = experiment_layout.split_manifest_path()
        ensure_file_exists(split_manifest_path, "split manifest")

        split_manifest = SplitManifest.read(split_manifest_path)

        if split_manifest.dataset_dir != str(self.dataset_dir.resolve()):
            raise ValueError(
                "split manifest dataset_dir does not match the training dataset"
            )
        
        ensure_fingerprint_matches(
            split_manifest.preprocessed_manifest_fingerprint,
            preprocessed_manifest,
            "split manifest was created from a different preprocessed manifest",
        )

        if not self.resume:
            return self

        train_manifest_path = experiment_layout.train_manifest_path()
        ensure_file_exists(train_manifest_path, "cannot resume: training manifest")

        train_manifest = TrainManifest.read(train_manifest_path)

        ensure_manifest_not_complete(
            train_manifest,
            "cannot resume: the training run is already complete",
        )
        
        ensure_training_resume_state(experiment_layout)
        
        return self

class InferConfig(FrozenModel):
    """Inference from a raw dataset using explicit checkpoints."""

    output_dir: Path = Field(description="Directory for inference artifacts.")
    dataset_dir: Path = Field(
        description="Indexed dataset directory containing the raw manifest.",
    )
    detector_checkpoint_path: Path = Field(
        description="Explicit detector checkpoint path.",
    )
    student_checkpoint_path: Path = Field(
        description="Explicit student checkpoint path.",
    )
    device: str = Field(
        description="Torch device string, e.g. 'cpu' or 'cuda'.",
    )

    @model_validator(mode="after")
    def validate_config(self) -> "InferConfig":
        ensure_device_available(self.device)
        for stage, checkpoint in (
            (DETECTOR_STAGE, self.detector_checkpoint_path),
            (STUDENT_STAGE, self.student_checkpoint_path),
        ):
            ensure_file_exists(checkpoint, f"{stage} checkpoint")

        ensure_directory_exists(self.dataset_dir, "dataset_dir")

        layout = DatasetLayout(dataset_dir=self.dataset_dir)
        manifest_path = layout.raw_manifest_path()
        ensure_file_exists(manifest_path, "raw manifest")

        RawDatasetManifest.read(manifest_path)
        return self


class EvaluateConfig(FrozenModel):
    dataset_dir: Path = Field(description="Dataset directory to evaluate.")
    experiment_dir: Path | None = Field(
        default=None,
        description="Existing experiment directory for held-out evaluation.",
    )
    output_dir: Path = Field(
        description="Output directory for explicit evaluation artifacts.",
    )
    detector_checkpoint_path: Path = Field(
        description="Detector checkpoint for explicit evaluation.",
    )
    student_checkpoint_path: Path = Field(
        description="Student checkpoint for explicit evaluation.",
    )
    device: str = Field(
        description="Torch device string, e.g. 'cpu' or 'cuda'.",
    )

    @model_validator(mode="before")
    @classmethod
    def derive_experiment_paths(cls, values: Any) -> Any:
        if not isinstance(values, dict) or values.get("experiment_dir") is None:
            return values

        experiment_layout = ExperimentLayout(
            experiment_dir=values["experiment_dir"]
        )
        return {
            **values,
            "output_dir": values["experiment_dir"],
            "detector_checkpoint_path": experiment_layout.best_checkpoint_path(
                DETECTOR_STAGE
            ),
            "student_checkpoint_path": experiment_layout.best_checkpoint_path(
                STUDENT_STAGE
            ),
        }

    @model_validator(mode="after")
    def validate_config(self) -> "EvaluateConfig":
        ensure_device_available(self.device)

        if self.experiment_dir is not None:
            experiment_layout = ExperimentLayout(experiment_dir=self.experiment_dir)

            ensure_directory_exists(self.dataset_dir, "dataset_dir")
            dataset_layout = DatasetLayout(dataset_dir=self.dataset_dir)
            train_manifest_path = experiment_layout.train_manifest_path()
            ensure_file_exists(train_manifest_path, "train manifest")
            train_manifest = TrainManifest.read(train_manifest_path)
            ensure_manifest_complete(
                train_manifest, train_manifest_path, "train manifest"
            )
            split_manifest_path = experiment_layout.split_manifest_path()
            ensure_file_exists(split_manifest_path, "split manifest")
            split_manifest = SplitManifest.read(split_manifest_path)
            ensure_manifest_complete(
                split_manifest, split_manifest_path, "split manifest"
            )
            preprocessed_manifest_path = dataset_layout.preprocessed_manifest_path()
            ensure_file_exists(preprocessed_manifest_path, "preprocessed manifest")
            preprocessed_manifest = PreprocessedDatasetManifest.read(
                preprocessed_manifest_path
            )
            ensure_manifest_complete(
                preprocessed_manifest,
                preprocessed_manifest_path,
                "preprocessed manifest",
            )
            ensure_fingerprint_matches(
                split_manifest.preprocessed_manifest_fingerprint,
                preprocessed_manifest,
                "split manifest was created from a different preprocessed manifest",
            )
            ensure_fingerprint_matches(
                train_manifest.split_manifest_fingerprint,
                split_manifest,
                "train manifest was created from a different split manifest",
            )
            return self

        for stage, checkpoint in (
            (DETECTOR_STAGE, self.detector_checkpoint_path),
            (STUDENT_STAGE, self.student_checkpoint_path),
        ):
            ensure_file_exists(checkpoint, f"{stage} checkpoint")
        ensure_directory_exists(self.dataset_dir, "dataset_dir")
        layout = DatasetLayout(dataset_dir=self.dataset_dir)
        manifest_path = layout.raw_manifest_path()
        ensure_file_exists(manifest_path, "raw manifest")
        manifest = RawDatasetManifest.read(manifest_path)
        ensure_subjects_have_masks(
            manifest.subjects,
            "cannot evaluate subjects without masks",
        )
        return self


class BasePatchConfig(FrozenModel):
    experiment_layout: ExperimentLayout = Field(
        description="Layout that owns materialized patch paths.",
    )
    stage: StageName = Field(
        description="Training stage owning the materialized patches.",
    )
    split: SplitName = Field(
        description="Dataset split owning the materialized patches.",
    )
    subjects: list[PreprocessedSubject] = Field(
        min_length=1,
        description="Preprocessed subjects from which patches are extracted.",
    )
    patch_size: int = Field(
        gt=0,
        description="Cubic patch edge length in voxels.",
    )
    augmentation_factor: int = Field(
        gt=0,
        description="Total records per patch, including one original record.",
    )


class NonOverlappingPatchConfig(BasePatchConfig):
    pass


class TargetCenteredPatchConfig(BasePatchConfig):
    probability_threshold: float = Field(
        ge=0,
        le=1,
        description="Minimum detector probability retained as a candidate.",
    )
    detector: Any = Field(
        description="Loaded detector used to locate candidate centers.",
    )

    @field_validator("detector")
    @classmethod
    def validate_detector(cls, detector: Any) -> Any:
        from ..core.common.models import CandidateDetector

        if not isinstance(detector, CandidateDetector):
            raise TypeError("detector must be a CandidateDetector")
        return detector
