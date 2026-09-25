import math
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from ..constants import (
    DETECTOR_STAGE,
    DEVICE_FIELD_DESCRIPTION,
    PREPROCESSED_MANIFEST_LABEL,
    RAW_MANIFEST_LABEL,
    SOURCE_ID_PATTERN,
    SOURCE_ID_PLACEHOLDER,
    SPLIT_MANIFEST_LABEL,
    STUDENT_STAGE,
    SUBJECT_ID_PLACEHOLDER,
    TEACHER_STAGE,
    TRAIN_MANIFEST_LABEL,
)
from ..core.datamodels import FrozenModel, Hyperparameters, Modality
from ..errors import ApplicationError
from .layouts import (
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
    """Require a configured path to reference an existing directory."""
    if not directory.is_dir():
        raise ApplicationError(
            category="Configuration",
            summary=f"Required {description} directory not found",
            fix="Create the directory or correct the configured path",
            context={"path": str(directory)},
        )


def ensure_file_exists(file_path: Path, description: str) -> None:
    """Require a configured path to reference an existing file."""
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
    """Reject an upstream manifest that has not completed successfully."""
    if manifest.status is not ManifestStatus.COMPLETE:
        raise ApplicationError(
            category="Manifest",
            summary=f"Cannot use incomplete {description} manifest",
            cause=(
                f"Manifest at '{manifest_path}' has status "
                f"'{manifest.status.value}'"
            ),
            fix="Rerun the producing stage to completion before continuing",
            context={"path": str(manifest_path)},
        )


def ensure_manifest_not_complete(manifest: Manifest) -> None:
    """Reject resume attempts for a pipeline stage that already completed."""
    if manifest.status is ManifestStatus.COMPLETE:
        raise ApplicationError(
            category="Manifest",
            summary="Cannot resume a completed manifest",
            cause="The producing stage is already complete",
            fix="Set resume to false to run the stage again",
        )


def ensure_training_resume_state(experiment_layout: ExperimentLayout) -> None:
    """Require a checkpoint or completed stage from which training can resume."""
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
    """Require every subject and preprocessed variant to provide a mask."""
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
            fix=(
                "Rebuild the indexed dataset with masks for every source, then "
                "rerun preprocessing if required"
            ),
        )


def ensure_fingerprint_matches(
    actual_fingerprint: str,
    expected_manifest: Manifest,
) -> None:
    """Require a stored fingerprint to match the current upstream manifest."""
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
            fix="Set resume to false and rerun preprocessing from the raw manifest",
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
                fix=(
                    "Set resume to false and rerun preprocessing from the raw "
                    "manifest"
                ),
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
                    fix=(
                        "Set resume to false and rerun preprocessing from the raw "
                        "manifest"
                    ),
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
        """Validate source naming patterns and input directory relationships."""
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
        """Validate raw inputs and resumable preprocessing state."""
        ensure_directory_exists(self.dataset_dir, "dataset_dir")

        layout = DatasetLayout(dataset_dir=self.dataset_dir)
        manifest_path = layout.raw_manifest_path()
        ensure_file_exists(manifest_path, RAW_MANIFEST_LABEL)

        raw_manifest = RawDatasetManifest.read(manifest_path)
        ensure_manifest_complete(raw_manifest, manifest_path, "raw")
        
        if not self.resume:
            return self

        preprocessed_manifest_path = layout.preprocessed_manifest_path()
        ensure_file_exists(
            preprocessed_manifest_path,
            PREPROCESSED_MANIFEST_LABEL,
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
        """Validate split proportions and the preprocessed dataset inputs."""
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
        ensure_file_exists(manifest_path, PREPROCESSED_MANIFEST_LABEL)

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
        description=DEVICE_FIELD_DESCRIPTION,
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
        """Validate training artifacts, augmentation limits, and resume state."""
        ensure_device_available(self.device)

        ensure_directory_exists(self.dataset_dir, "dataset_dir")

        dataset_layout = DatasetLayout(dataset_dir=self.dataset_dir)
        preprocessed_manifest_path = dataset_layout.preprocessed_manifest_path()
        ensure_file_exists(preprocessed_manifest_path, PREPROCESSED_MANIFEST_LABEL)

        preprocessed_manifest = PreprocessedDatasetManifest.read(
            preprocessed_manifest_path
        )

        ensure_manifest_complete(
            preprocessed_manifest,
            preprocessed_manifest_path,
            "manifest",
        )
        requested_factors = {
            "detector": self.detector_augmentation_factor,
            "discriminator": self.discriminator_augmentation_factor,
        }
        unavailable_factors = [
            f"{name}={factor}"
            for name, factor in requested_factors.items()
            if factor > preprocessed_manifest.augmentation_factor
        ]
        if unavailable_factors:
            raise ApplicationError(
                category="Configuration",
                summary="Training augmentation factor exceeds preprocessed output",
                cause=(
                    f"Requested {', '.join(unavailable_factors)}, but preprocessing "
                    f"produced {preprocessed_manifest.augmentation_factor} variants "
                    "per subject"
                ),
                fix=(
                    "Reduce the training augmentation factors or rerun preprocessing "
                    "with a larger augmentation factor"
                ),
                context={"path": str(preprocessed_manifest_path)},
            )

        experiment_layout = ExperimentLayout(experiment_dir=self.experiment_dir)
        split_manifest_path = experiment_layout.split_manifest_path()
        ensure_file_exists(split_manifest_path, SPLIT_MANIFEST_LABEL)

        split_manifest = SplitManifest.read(split_manifest_path)
        ensure_manifest_complete(split_manifest, split_manifest_path, "split")

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
        ensure_file_exists(train_manifest_path, TRAIN_MANIFEST_LABEL)

        train_manifest = TrainManifest.read(train_manifest_path)

        ensure_manifest_not_complete(
            train_manifest,
        )

        ensure_fingerprint_matches(
            train_manifest.split_manifest_fingerprint,
            split_manifest,
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
        description=DEVICE_FIELD_DESCRIPTION,
    )

    @model_validator(mode="after")
    def validate_config(self) -> "InferConfig":
        """Validate inference checkpoints, device, and raw dataset inputs."""
        ensure_device_available(self.device)
        for stage, checkpoint in (
            (DETECTOR_STAGE, self.detector_checkpoint_path),
            (STUDENT_STAGE, self.student_checkpoint_path),
        ):
            ensure_file_exists(checkpoint, f"{stage} checkpoint")

        ensure_directory_exists(self.dataset_dir, "dataset_dir")

        layout = DatasetLayout(dataset_dir=self.dataset_dir)
        manifest_path = layout.raw_manifest_path()
        ensure_file_exists(manifest_path, RAW_MANIFEST_LABEL)

        raw_manifest = RawDatasetManifest.read(manifest_path)
        ensure_manifest_complete(raw_manifest, manifest_path, "raw")
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
        description=DEVICE_FIELD_DESCRIPTION,
    )

    @model_validator(mode="before")
    @classmethod
    def derive_experiment_paths(cls, values: Any) -> Any:
        """Derive evaluation outputs and checkpoints from an experiment directory."""
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
        """Validate explicit or experiment-backed evaluation artifacts."""
        ensure_device_available(self.device)

        if self.experiment_dir is not None:
            experiment_layout = ExperimentLayout(experiment_dir=self.experiment_dir)

            ensure_directory_exists(self.dataset_dir, "dataset_dir")
            dataset_layout = DatasetLayout(dataset_dir=self.dataset_dir)
            train_manifest_path = experiment_layout.train_manifest_path()
            ensure_file_exists(train_manifest_path, TRAIN_MANIFEST_LABEL)
            train_manifest = TrainManifest.read(train_manifest_path)
            ensure_manifest_complete(
                train_manifest, train_manifest_path, TRAIN_MANIFEST_LABEL
            )
            split_manifest_path = experiment_layout.split_manifest_path()
            ensure_file_exists(split_manifest_path, SPLIT_MANIFEST_LABEL)
            split_manifest = SplitManifest.read(split_manifest_path)
            ensure_manifest_complete(
                split_manifest, split_manifest_path, SPLIT_MANIFEST_LABEL
            )
            preprocessed_manifest_path = dataset_layout.preprocessed_manifest_path()
            ensure_file_exists(
                preprocessed_manifest_path, PREPROCESSED_MANIFEST_LABEL
            )
            preprocessed_manifest = PreprocessedDatasetManifest.read(
                preprocessed_manifest_path
            )
            ensure_manifest_complete(
                preprocessed_manifest,
                preprocessed_manifest_path,
                PREPROCESSED_MANIFEST_LABEL,
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
        ensure_file_exists(manifest_path, RAW_MANIFEST_LABEL)
        manifest = RawDatasetManifest.read(manifest_path)
        ensure_manifest_complete(manifest, manifest_path, "raw")
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

    @model_validator(mode="after")
    def validate_inputs(self) -> "BasePatchConfig":
        """Validate patch variants, masks, and source artifacts for extraction."""
        for subject in self.subjects:
            if len(subject.variants) < self.augmentation_factor:
                raise ApplicationError(
                    category="Configuration",
                    summary="Patch augmentation factor exceeds available variants",
                    cause=(
                        f"Subject '{subject.subject_id}' has "
                        f"{len(subject.variants)} variants, but "
                        f"{self.augmentation_factor} were requested"
                    ),
                    fix=(
                        "Reduce the patch augmentation factor or rerun preprocessing "
                        "with more variants"
                    ),
                    context={"subject_id": subject.subject_id},
                )

            for variant_index, variant in enumerate(
                subject.variants[: self.augmentation_factor]
            ):
                if variant.mask_path is None:
                    raise ApplicationError(
                        category="Input data",
                        summary="Patch extraction requires a variant mask",
                        cause=(
                            f"Subject '{subject.subject_id}' variant "
                            f"{variant_index} has no mask"
                        ),
                        fix=(
                            "Complete preprocessing with masks before extracting "
                            "training patches"
                        ),
                        context={"subject_id": subject.subject_id},
                    )

                input_paths = (
                    ("volume", variant.volume_path),
                    ("mask", variant.mask_path),
                    ("FRST", variant.frst_path),
                )
                for input_name, input_path in input_paths:
                    if not Path(input_path).is_file():
                        raise ApplicationError(
                            category="Input data",
                            summary="Required patch input file is missing",
                            cause=(
                                f"Subject '{subject.subject_id}' variant "
                                f"{variant_index} {input_name} file was not found"
                            ),
                            fix=(
                                "Restore the preprocessed output or rerun "
                                "preprocessing"
                            ),
                            context={"path": str(input_path)},
                        )
        return self


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
        """Require the detector implementation used for candidate localization."""
        from ..core.common.models import CandidateDetector

        if not isinstance(detector, CandidateDetector):
            raise TypeError(
                "Target-centered patch extraction requires a CandidateDetector "
                f"instance, received {type(detector).__name__}"
            )
        return detector
