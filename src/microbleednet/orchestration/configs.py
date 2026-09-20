"""Top-level command configs the orchestration layer consumes.

Each pipe's ``execute()`` takes one of these config objects whole. They live
here because the orchestration layer owns them; the CLI imports them to parse
TOML into already-validated objects, which keeps the dependency direction
``cli -> orchestration`` intact. Loading a config from a TOML file is a CLI
concern (``cli/utils.py``).
"""

import re
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from ..core.datamodels import FrozenModel, Hyperparameters, Modality
from .layouts import (
    DETECTOR_STAGE,
    STUDENT_STAGE,
    DatasetLayout,
    ExperimentLayout,
)
from .manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    RawDatasetManifest,
    SplitManifest,
    TrainManifest,
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
DEVICE_DESCRIPTION = "Torch device string, e.g. 'cpu' or 'cuda'."


def ensure_device_available(device_name: str) -> None:
    """Validate a device string without importing torch at module load time."""
    import torch

    try:
        device = torch.device(device_name)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid device: {device_name}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device is not available")


def require_dataset_dir(
    dataset_dir: Path, manifest_path: Path, manifest_name: str
) -> None:
    if not dataset_dir.is_dir():
        raise ValueError(f"dataset_dir does not exist: {dataset_dir}")
    if not manifest_path.is_file():
        raise ValueError(f"{manifest_name} does not exist: {manifest_path}")


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
            "Use it to keep subjects unique when "
            "indexing several sources into one dataset. "
            "Allowed characters: letters, digits, '-', '_'."
        ),
    )
    modality: Modality = Field(
        description="Imaging modality of this source's volumes.",
    )

    @model_validator(mode="after")
    def validate_source_id(self) -> "IndexDataConfig":
        if not SOURCE_ID_PATTERN.match(self.source_id):
            raise ValueError(
                f"{SOURCE_ID_PLACEHOLDER} may contain only letters, digits, "
                "'-', and '_'"
            )
        return self

    @model_validator(mode="after")
    def validate_patterns(self) -> "IndexDataConfig":
        patterns = [("volume_pattern", self.volume_pattern)]
        if self.mask_pattern is not None:
            patterns.append(("mask_pattern", self.mask_pattern))
        for name, pattern in patterns:
            if pattern.count(SUBJECT_ID_PLACEHOLDER) < 1:
                raise ValueError(
                    f"{name} must contain the '{SUBJECT_ID_PLACEHOLDER}' "
                    "placeholder at least once"
                )
        return self

    @model_validator(mode="after")
    def validate_directories(self) -> "IndexDataConfig":
        if not self.input_dir.is_dir():
            raise ValueError(f"input_dir does not exist: {self.input_dir}")
        if (self.mask_dir is None) != (self.mask_pattern is None):
            raise ValueError("mask_dir and mask_pattern must be provided together")
        if self.mask_dir is not None and not self.mask_dir.is_dir():
            raise ValueError(f"mask_dir does not exist: {self.mask_dir}")
        return self


class PreprocessConfig(FrozenModel):
    dataset_dir: Path = Field(
        description="Indexed dataset directory containing manifests/raw.json."
    )
    augmentation_factor: int = Field(
        gt=0,
        description=(
            "Total persisted variants per subject, including the original."
        ),
    )
    resume: bool = Field(
        default=False,
        description="Resume from the last per-subject preprocessing checkpoint.",
    )

    @model_validator(mode="after")
    def validate_dataset_dir(self) -> "PreprocessConfig":
        layout = DatasetLayout(dataset_dir=self.dataset_dir)
        manifest_path = layout.raw_manifest_path()
        require_dataset_dir(self.dataset_dir, manifest_path, "raw manifest")
        manifest = RawDatasetManifest.read(manifest_path)
        preprocessed_manifest_path = layout.preprocessed_manifest_path()
        if not preprocessed_manifest_path.exists():
            return self

        if not self.resume:
            return self

        existing_manifest = PreprocessedDatasetManifest.read(preprocessed_manifest_path)
        if existing_manifest.status is ManifestStatus.RUNNING:
            raise ValueError(
                "a running preprocessing checkpoint exists; use resume=true "
                "to continue it"
            )
        if existing_manifest.raw_manifest_fingerprint != content_fingerprint(manifest):
            raise ValueError("cannot resume: the raw dataset manifest has changed")
        if existing_manifest.augmentation_factor != self.augmentation_factor:
            raise ValueError("cannot resume: the augmentation factor has changed")
        return self


class SplitConfig(FrozenModel):
    dataset_dir: Path = Field(
        description="Preprocessed dataset directory containing subjects to split."
    )
    experiment_dir: Path = Field(
        description="Experiment directory to receive the split manifest."
    )
    train_size: float = Field(
        default=0.7,
        gt=0,
        le=1,
        description="Proportion of subjects assigned to the training split.",
    )
    validation_size: float = Field(
        default=0.1,
        gt=0,
        le=1,
        description="Proportion of subjects assigned to the validation split.",
    )
    test_size: float = Field(
        default=0.2,
        gt=0,
        le=1,
        description="Proportion of subjects held out for testing.",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description="Optional random seed for reproducible subject splitting.",
    )

    @model_validator(mode="after")
    def validate_split_sizes(self) -> "SplitConfig":
        split_total = self.train_size + self.validation_size + self.test_size
        if abs(split_total - 1.0) > 1e-9:
            raise ValueError(
                "train_size, validation_size, and test_size must sum to 1.0"
            )
        return self

    @model_validator(mode="after")
    def validate_dataset(self) -> "SplitConfig":
        manifest_path = DatasetLayout(
            dataset_dir=self.dataset_dir
        ).preprocessed_manifest_path()
        require_dataset_dir(self.dataset_dir, manifest_path, "preprocessed manifest")
        manifest = PreprocessedDatasetManifest.read(manifest_path)
        if manifest.status is not ManifestStatus.COMPLETE:
            raise ValueError(
                f"manifest at {manifest_path} has status {manifest.status.value!r}; "
                "a consumer may only read a complete manifest."
            )
        maskless_subjects = [
            subject.subject_id
            for subject in manifest.subjects
            if any(variant.mask_path is None for variant in subject.variants)
        ]
        if maskless_subjects:
            raise ValueError(
                "cannot split a preprocessed dataset with subjects missing masks: "
                + ", ".join(maskless_subjects)
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
        default="cpu",
        description=DEVICE_DESCRIPTION,
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description="Optional random seed for reproducible training runs.",
    )
    detector_candidate_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Minimum detector probability retained as a student candidate.",
    )
    detector_augmentation_factor: int = Field(
        default=10,
        gt=0,
        description=(
            "Total variants used for detector training, including the original."
        ),
    )
    discriminator_augmentation_factor: int = Field(
        default=5,
        gt=0,
        description=(
            "Total variants used for discriminator training, including the original."
        ),
    )
    num_workers: int = Field(
        default=0,
        ge=0,
        description="Number of worker processes used by training data loaders.",
    )
    pin_memory: bool = Field(
        default=False,
        description=(
            "Whether training data loaders use page-locked host memory for transfers."
        ),
    )
    detector_hyperparameters: Hyperparameters = Field(
        default_factory=Hyperparameters,
        description="Optimizer and training-loop settings for detector training.",
    )
    teacher_hyperparameters: Hyperparameters = Field(
        default_factory=Hyperparameters,
        description="Optimizer and training-loop settings for teacher training.",
    )
    student_hyperparameters: Hyperparameters = Field(
        default_factory=Hyperparameters,
        description="Optimizer and training-loop settings for student training.",
    )

    @model_validator(mode="after")
    def validate_device(self) -> "TrainConfig":
        ensure_device_available(self.device)
        return self

    @model_validator(mode="after")
    def validate_dataset(self) -> "TrainConfig":
        manifest_path = DatasetLayout(
            dataset_dir=self.dataset_dir
        ).preprocessed_manifest_path()
        require_dataset_dir(self.dataset_dir, manifest_path, "preprocessed manifest")
        manifest = PreprocessedDatasetManifest.read(manifest_path)
        if manifest.status is not ManifestStatus.COMPLETE:
            raise ValueError(
                f"manifest at {manifest_path} has status {manifest.status.value!r}; "
                "a consumer may only read a complete manifest."
            )
        return self

    @model_validator(mode="after")
    def validate_split_manifest_dataset(self) -> "TrainConfig":
        split_manifest_path = ExperimentLayout(
            experiment_dir=self.experiment_dir
        ).split_manifest_path()
        if not split_manifest_path.is_file():
            raise ValueError(f"split manifest does not exist: {split_manifest_path}")
        split_manifest = SplitManifest.read(split_manifest_path)
        if split_manifest.dataset_dir != str(self.dataset_dir.resolve()):
            raise ValueError(
                "split manifest dataset_dir does not match the training dataset"
            )
        return self


class InferConfig(FrozenModel):
    """Configuration for the internal preprocessed-subject inference pipe."""

    subjects: list[PreprocessedSubject] = Field(
        min_length=1,
        description="Preprocessed subjects to infer.",
    )
    experiment_dir: Path = Field(
        description="Experiment directory containing detector and student checkpoints.",
    )
    device: str = Field(
        default="cpu",
        description=DEVICE_DESCRIPTION,
    )

    @model_validator(mode="after")
    def validate_device(self) -> "InferConfig":
        ensure_device_available(self.device)
        return self

    @model_validator(mode="after")
    def validate_checkpoints(self) -> "InferConfig":
        layout = ExperimentLayout(experiment_dir=self.experiment_dir)
        for stage in (DETECTOR_STAGE, STUDENT_STAGE):
            checkpoint = layout.best_checkpoint_path(stage)
            if not checkpoint.is_file():
                raise ValueError(f"{stage} checkpoint does not exist: {checkpoint}")
        return self


class EvaluateConfig(FrozenModel):
    """Configuration for held-out final-pipeline evaluation."""

    experiment_dir: Path = Field(
        description="Experiment directory containing train and inference artifacts."
    )
    dataset_dir: Path = Field(
        description="Indexed dataset directory containing preprocessed subjects."
    )
    device: str = Field(
        default="cpu",
        description=DEVICE_DESCRIPTION,
    )

    @model_validator(mode="after")
    def validate_device(self) -> "EvaluateConfig":
        ensure_device_available(self.device)
        return self

    @model_validator(mode="after")
    def validate_manifests(self) -> "EvaluateConfig":
        experiment_layout = ExperimentLayout(experiment_dir=self.experiment_dir)
        dataset_layout = DatasetLayout(dataset_dir=self.dataset_dir)
        manifest_paths = (
            (experiment_layout.train_manifest_path(), TrainManifest, "train manifest"),
            (experiment_layout.split_manifest_path(), SplitManifest, "split manifest"),
            (
                dataset_layout.preprocessed_manifest_path(),
                PreprocessedDatasetManifest,
                "preprocessed manifest",
            ),
        )
        for manifest_path, manifest_type, manifest_name in manifest_paths:
            require_dataset_dir(self.dataset_dir, manifest_path, manifest_name)
            manifest = manifest_type.read(manifest_path)
            if manifest.status is not ManifestStatus.COMPLETE:
                raise ValueError(
                    f"manifest at {manifest_path} has status "
                    f"{manifest.status.value!r}; a consumer may only read a "
                    "complete manifest."
                )
        return self


class BasePatchConfig(FrozenModel):
    experiment_layout: ExperimentLayout = Field(
        description="Layout that owns materialized patch paths.",
    )
    stage: str = Field(
        description="Training stage owning the materialized patches.",
    )
    split: str = Field(
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

