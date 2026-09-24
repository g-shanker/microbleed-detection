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
from ..errors import ApplicationError
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
        raise ApplicationError(
            category="Configuration",
            summary="Invalid compute device",
            cause=f"Torch could not interpret '{device_name}'",
            fix="Use a valid device such as 'cpu' or a configured CUDA device",
            context={"device": device_name},
        ) from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ApplicationError(
            category="Environment",
            summary="Requested CUDA device is unavailable",
            cause=(
                f"The configured device is '{device_name}', but CUDA is unavailable"
            ),
            fix=(
                "Use 'cpu' or install and configure a CUDA-capable "
                "PyTorch environment"
            ),
            context={"device": device_name},
        )


def ensure_directory_exists(directory: Path, description: str) -> None:
    if not directory.is_dir():
        raise ApplicationError(
            category="Configuration",
            summary=f"Required {description} directory not found",
            fix="Create the directory or correct the configured path",
            context={"path": str(directory)},
        )


def ensure_file_exists(file_path: Path, description: str) -> None:
    if not file_path.is_file():
        raise ApplicationError(
            category="Configuration",
            summary=f"Required {description} file not found",
            fix="Run the producing stage or correct the configured path",
            context={"path": str(file_path)},
        )


def ensure_manifest_complete(
    manifest: Manifest, manifest_path: Path, description: str
) -> None:
    if manifest.status is not ManifestStatus.COMPLETE:
        raise ApplicationError(
            category="Manifest",
            summary=f"Cannot use incomplete {description} manifest",
            cause=f"Manifest status is '{manifest.status.value}'",
            fix="Complete or resume the producing stage before continuing",
            context={"path": str(manifest_path)},
        )


def ensure_manifest_not_complete(manifest: Manifest) -> None:
    if manifest.status is ManifestStatus.COMPLETE:
        raise ApplicationError(
            category="Manifest",
            summary="Cannot resume a completed manifest",
            cause="The producing stage is already complete",
            fix="Disable resume or select a new output location",
        )


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

    raise ApplicationError(
        category="Checkpoint",
        summary="No recoverable training state found",
        cause="No stage checkpoint or completed stage manifest is available",
        fix="Disable resume or restore a checkpoint or completed stage manifest",
    )


def ensure_subjects_have_masks(
    subjects: list[RawSubject] | list[PreprocessedSubject],
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
        displayed_ids = ", ".join(maskless_subjects[:5])
        if len(maskless_subjects) > 5:
            displayed_ids += f", ... ({len(maskless_subjects)} total)"
        raise ApplicationError(
            category="Input data",
            summary="Required subject masks are missing",
            cause=f"Subjects without masks: {displayed_ids}",
            fix="Provide masks for every subject in this workflow",
        )


def ensure_fingerprint_matches(
    actual_fingerprint: str,
    expected_manifest: Manifest,
) -> None:
    if actual_fingerprint != content_fingerprint(expected_manifest):
        raise ApplicationError(
            category="Manifest",
            summary="Manifest fingerprint does not match",
            cause=(
                "The upstream artifact changed after the downstream artifact "
                "was created"
            ),
            fix=(
                "Regenerate downstream artifacts or restore the matching "
                "upstream manifest"
            ),
        )


def ensure_preprocessed_subjects_complete(
    subjects: list[PreprocessedSubject],
    raw_subject_ids: set[str],
    augmentation_factor: int,
) -> None:
    """Validate subjects already persisted in a resumable preprocessing run."""
    unexpected_ids = sorted(
        subject.subject_id
        for subject in subjects
        if subject.subject_id not in raw_subject_ids
    )
    if unexpected_ids:
        displayed_ids = ", ".join(unexpected_ids[:5])
        if len(unexpected_ids) > 5:  # pragma: no branch
            displayed_ids += f", ... ({len(unexpected_ids)} total)"
        raise ApplicationError(
            category="Manifest",
            summary="Preprocessed manifest contains unknown subjects",
            cause=f"Subject IDs are not present in the raw manifest: {displayed_ids}",
            fix="Remove the stale preprocessed manifest and rerun preprocessing",
        )

    for subject in subjects:  # pragma: no branch
        if len(subject.variants) != augmentation_factor:
            raise ApplicationError(
                category="Manifest",
                summary="Preprocessed subject has incomplete variants",
                cause=(
                    f"Subject '{subject.subject_id}' has {len(subject.variants)} "
                    f"variants, but {augmentation_factor} are required"
                ),
                fix="Remove the incomplete subject output and rerun preprocessing",
                context={"subject_id": subject.subject_id},
            )
        for variant_index, variant in enumerate(subject.variants):  # pragma: no branch
            paths = (variant.volume_path, variant.frst_path, variant.mask_path)
            missing_paths = [
                path
                for path in paths  # pragma: no branch
                if path is not None and not Path(path).is_file()  # pragma: no branch
            ]
            if missing_paths:
                raise ApplicationError(
                    category="Input data",
                    summary="Preprocessed variant output is missing",
                    cause=(
                        f"Subject '{subject.subject_id}' variant {variant_index} "
                        f"references missing files: {', '.join(missing_paths)}"
                    ),
                    fix="Remove the incomplete subject output and rerun preprocessing",
                    context={"subject_id": subject.subject_id},
                )


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
            raise ApplicationError(
                category="Configuration",
                summary="Source ID is not filesystem-safe",
                cause=(
                    f"'{self.source_id}' contains characters outside letters, "
                    "digits, '-' and '_'"
                ),
                fix="Use a source ID containing only letters, digits, '-' and '_'",
                context={"field": "source_id"},
            )

        patterns = [("volume_pattern", self.volume_pattern)]
        if self.mask_pattern is not None:
            patterns.append(("mask_pattern", self.mask_pattern))
        for name, pattern in patterns:
            if pattern.count(SUBJECT_ID_PLACEHOLDER) < 1:
                raise ApplicationError(
                    category="Configuration",
                    summary=f"{name} is missing the subject ID placeholder",
                    cause=f"The pattern must contain '{SUBJECT_ID_PLACEHOLDER}'",
                    fix=f"Add '{SUBJECT_ID_PLACEHOLDER}' to the {name} pattern",
                    context={"field": name},
                )

        ensure_directory_exists(self.input_dir, "input_dir")
        if self.mask_dir is not None:
            ensure_directory_exists(self.mask_dir, "mask_dir")
        if (self.mask_dir is None) != (self.mask_pattern is None):
            raise ApplicationError(
                category="Configuration",
                summary="Mask directory and mask pattern must be provided together",
                cause="One mask setting is present without the other",
                fix="Provide both mask_dir and mask_pattern, or omit both",
            )

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
            "preprocessed manifest",
        )

        existing_manifest = PreprocessedDatasetManifest.read(preprocessed_manifest_path)

        ensure_manifest_not_complete(
            existing_manifest,
        )

        ensure_fingerprint_matches(
            existing_manifest.raw_manifest_fingerprint,
            raw_manifest,
        )
        ensure_preprocessed_subjects_complete(
            existing_manifest.subjects,
            {subject.subject_id for subject in raw_manifest.subjects},
            self.augmentation_factor,
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
            raise ApplicationError(
                category="Configuration",
                summary="Split proportions must total 1.0",
                cause=f"Received proportions total {split_total:g}",
                fix=(
                    "Adjust train_size, validation_size and test_size so they "
                    "total 1.0"
                ),
            )

        ensure_directory_exists(self.dataset_dir, "dataset_dir")

        layout = DatasetLayout(dataset_dir=self.dataset_dir)
        manifest_path = layout.preprocessed_manifest_path()
        ensure_file_exists(manifest_path, "preprocessed manifest")

        manifest = PreprocessedDatasetManifest.read(manifest_path)
        
        ensure_manifest_complete(
            manifest,
            manifest_path,
            "preprocessed",
        )

        ensure_subjects_have_masks(
            manifest.subjects,
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
            raise ApplicationError(
                category="Manifest",
                summary="Split manifest belongs to a different dataset",
                cause=(
                    f"Manifest dataset is '{split_manifest.dataset_dir}', "
                    f"but configuration uses '{self.dataset_dir.resolve()}'"
                ),
                fix="Use the matching dataset and split manifest",
            )
        
        ensure_fingerprint_matches(
            split_manifest.preprocessed_manifest_fingerprint,
            preprocessed_manifest,
        )

        if not self.resume:
            return self

        train_manifest_path = experiment_layout.train_manifest_path()
        ensure_file_exists(train_manifest_path, "train manifest")

        train_manifest = TrainManifest.read(train_manifest_path)

        ensure_manifest_not_complete(
            train_manifest,
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
            )
            ensure_fingerprint_matches(
                train_manifest.split_manifest_fingerprint,
                split_manifest,
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
            raise TypeError(
                "Target-centered patch extraction requires a CandidateDetector "
                f"instance, received {type(detector).__name__}"
            )
        return detector
