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

from ..core.datamodels import FrozenModel, Modality
from .layouts import DatasetLayout
from .manifests import (
    PreprocessedSubject,
    RawDatasetManifest,
)

# Token a volume/mask filename pattern must contain at least once; the text it
# matches becomes the subject ID. Shared by the index-data pipeline (which
# splits filenames on it) and IndexDataConfig (which validates its presence).
SUBJECT_ID_PLACEHOLDER = "{subject_id}"
SOURCE_ID_PLACEHOLDER = "{source_id}"

# A source_id namespaces subject IDs and becomes part of on-disk paths, so it is
# restricted to filesystem-safe characters.
SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


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
            f"Must contain the '{SUBJECT_ID_PLACEHOLDER}' placeholder."
        ),
    )
    label_dir: Path = Field(
        description="Label directory containing the masks to index.",
    )
    mask_pattern: str = Field(
        description=(
            "Naming pattern of masks in the label directory. "
            f"Must contain the '{SUBJECT_ID_PLACEHOLDER}' placeholder."
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
        default="T2*-GRE",
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
        for name, pattern in (
            ("volume_pattern", self.volume_pattern),
            ("mask_pattern", self.mask_pattern),
        ):
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
        if not self.label_dir.is_dir():
            raise ValueError(f"label_dir does not exist: {self.label_dir}")
        return self


class PreprocessConfig(FrozenModel):
    dataset_dir: Path = Field(
        description="Indexed dataset directory containing manifests/raw.json."
    )

    @model_validator(mode="after")
    def validate_dataset_dir(self) -> "PreprocessConfig":
        if not self.dataset_dir.is_dir():
            raise ValueError(f"dataset_dir does not exist: {self.dataset_dir}")
        manifest_path = DatasetLayout(dataset_dir=self.dataset_dir).raw_manifest_path()
        if not manifest_path.is_file():
            raise ValueError(f"raw manifest does not exist: {manifest_path}")
        RawDatasetManifest.read(manifest_path)
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
        description="Torch device string, e.g. 'cpu' or 'cuda'.",
    )

    @model_validator(mode="after")
    def validate_device(self) -> "TrainConfig":
        import torch

        try:
            device = torch.device(self.device)
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"invalid device: {self.device}") from error
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device is not available")
        return self

    @model_validator(mode="after")
    def validate_dataset(self) -> "TrainConfig":
        if not self.dataset_dir.is_dir():
            raise ValueError(f"dataset_dir does not exist: {self.dataset_dir}")
        manifest_path = DatasetLayout(
            dataset_dir=self.dataset_dir
        ).preprocessed_manifest_path()
        if not manifest_path.is_file():
            raise ValueError(f"preprocessed manifest does not exist: {manifest_path}")
        return self


class BasePatchConfig(FrozenModel):
    patch_dir: Path = Field(
        description="Directory where materialized patches are written.",
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

