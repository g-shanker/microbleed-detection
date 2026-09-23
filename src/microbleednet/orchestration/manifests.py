"""Manifests the orchestration layer writes and reads, with their durability contract.

A manifest is a typed, versioned record a pipe stage writes to advertise
what it produced and whether it finished. Every manifest shares one envelope — a
schema version, a stable type discriminator, a lifecycle status, and
creation/update timestamps — wrapped around a payload typed for that stage.

Rules enforced here (see ``ARCHITECTURE.md``):

- Every manifest carries ``manifest_type``, ``status``, ``created_at`` and
    ``updated_at``.
- Consumers must check that a manifest is ``complete`` before using its
    published outputs.
- Models forbid unknown fields, are immutable, and never type a field as
  ``Any``.

The models and their read/write helpers live together because the behavior is
intrinsic to the type and layer-agnostic: the helpers depend only on
:mod:`microbleednet.core.io`, so manifest persistence does not
pull in the ML stack.
"""

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from ..core import io as core_io
from ..core.datamodels import (
    BoundingBox,
    EpochLoss,
    EvaluationAggregate,
    EvaluationMetrics,
    FrozenModel,
    Hyperparameters,
    Modality,
    PatchRecord,
)
from .layouts import SplitName, StageName


def timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for manifest lifecycle fields."""
    return datetime.now(timezone.utc).isoformat()


class ManifestStatus(str, Enum):
    """Lifecycle state of the stage that produced a manifest."""

    RUNNING = "running"
    COMPLETE = "complete"


class Manifest(FrozenModel):
    """Envelope shared by every manifest."""

    status: ManifestStatus = Field(
        description="Whether the producing stage is running or complete.",
    )
    created_at: str = Field(
        default_factory=timestamp,
        description="ISO-8601 UTC time the manifest was created.",
    )
    updated_at: str = Field(
        default_factory=timestamp,
        description="ISO-8601 UTC time of the last write.",
    )

    def write(self, path: Path) -> None:
        """Serialize this manifest to ``path`` atomically as versioned JSON."""
        payload = self.model_dump(mode="json")

        if path.exists():
            existing = core_io.read_json(path)
            payload["created_at"] = existing.get("created_at", timestamp())

        payload["updated_at"] = timestamp()

        core_io.write_json(path, payload)


    @classmethod
    def read[ManifestType: Manifest](
        cls: type[ManifestType], path: Path
    ) -> ManifestType:
        """Load and validate a manifest from ``path``."""
        payload = core_io.read_json(path)

        try:
            manifest = cls.model_validate(payload)
        except ValidationError as error:
            raise ValueError(f"invalid manifest at {path}:\n{error}") from error
        
        return manifest


def content_fingerprint(manifest: Manifest) -> str:
    """Return a stable hash of manifest content excluding lifecycle fields."""
    payload = manifest.model_dump(
        mode="json",
        exclude={"status", "created_at", "updated_at"},
    )
    canonical_payload = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical_payload).hexdigest()


class RawSubject(FrozenModel):
    """One indexed subject: a volume and its required lesion mask."""

    subject_id: str = Field(description="Unique subject identifier.")
    source_id: str = Field(
        description="Identifier of the source that contributed this subject.",
    )
    volume_path: str = Field(description="Absolute path to the raw volume.")
    mask_path: str | None = Field(
        default=None, description="Absolute path to the lesion mask, when provided."
    )


class RawSource(FrozenModel):
    """A directory pair and filename patterns that contributed subjects."""

    input_dir: str = Field(description="Directory the volumes were indexed from.")
    mask_dir: str | None = Field(
        default=None, description="Directory the mask volumes were indexed from."
    )
    volume_pattern: str = Field(description="Glob/regex pattern matching volumes.")
    mask_pattern: str | None = Field(
        default=None, description="Pattern matching masks, when provided."
    )
    source_id: str = Field(
        description="Namespace prepended to this source's subject IDs.",
    )
    modality: Modality = Field(
        default="T2*-GRE",
        description="Imaging modality used to select preprocessing operations.",
    )
    added_on: str = Field(
        default_factory=timestamp,
        description="ISO-8601 time this source was indexed.",
    )


class RawDatasetManifest(Manifest):
    """Manifest ``index-data`` writes after building the raw dataset."""

    manifest_type: Literal["raw_dataset"] = Field(
        default="raw_dataset",
        description="Stable discriminator for an indexed raw dataset manifest.",
    )
    sources: list[RawSource] = Field(description="Sources that contributed subjects.")
    subjects: list[RawSubject] = Field(description="Indexed subjects.")

    @model_validator(mode="after")
    def unique_subjects(self) -> "RawDatasetManifest":
        reject_duplicate_ids(
            (subject.subject_id for subject in self.subjects), "subject"
        )
        return self

    @model_validator(mode="after")
    def unique_sources(self) -> "RawDatasetManifest":
        reject_duplicate_ids((source.source_id for source in self.sources), "source")
        return self


class PreprocessedSubject(FrozenModel):
    """One subject produced by the preprocessing stage."""

    subject_id: str = Field(description="Unique subject identifier.")
    original_volume_path: str = Field(
        description="Absolute path to the original volume for restoring inference.",
    )
    bounding_box: BoundingBox = Field(
        description="Crop bounds applied during preprocessing.",
    )
    variants: list["PreprocessedVariant"] = Field(
        description="Ordered preprocessed variants, including the original.",
    )


class PreprocessedVariant(FrozenModel):
    """One persisted volume, optional mask, and FRST result for a subject."""

    volume_path: str = Field(description="Absolute path to the variant volume.")
    mask_path: str | None = Field(
        default=None,
        description="Absolute path to the variant mask, when available.",
    )
    frst_path: str = Field(description="Absolute path to the variant FRST volume.")


class PreprocessedDatasetManifest(Manifest):
    """Manifest ``preprocess`` writes after preparing every subject."""

    manifest_type: Literal["preprocessed_dataset"] = Field(
        default="preprocessed_dataset",
        description="Stable discriminator for a preprocessed dataset manifest.",
    )
    subjects: list[PreprocessedSubject] = Field(
        description="Subjects produced by the preprocessing stage."
    )
    augmentation_factor: int = Field(
        default=1,
        gt=0,
        description="Total persisted variants per subject, including the original.",
    )
    raw_manifest_fingerprint: str = Field(
        description="Fingerprint of the raw manifest used to create this dataset.",
    )


class SplitManifest(Manifest):
    """Manifest ``split`` writes before training begins."""

    manifest_type: Literal["split"] = Field(
        default="split",
        description="Stable discriminator for a subject split manifest.",
    )
    dataset_dir: str = Field(
        description="Absolute path to the preprocessed dataset that was split."
    )
    preprocessed_manifest_fingerprint: str = Field(
        description="Fingerprint of the preprocessed manifest used for this split."
    )
    seed: int | None = Field(
        default=None, ge=0, description="Random seed used for subject splitting."
    )
    train_size: float = Field(
        gt=0, le=1, description="Ratio of subjects assigned to training."
    )
    validation_size: float = Field(
        gt=0, le=1, description="Ratio of subjects assigned to validation."
    )
    test_size: float = Field(
        gt=0, le=1, description="Ratio of subjects held out for testing."
    )
    train_subject_ids: list[str] = Field(
        description="Ordered subject IDs assigned to training."
    )
    validation_subject_ids: list[str] = Field(
        description="Ordered subject IDs assigned to validation."
    )
    test_subject_ids: list[str] = Field(
        description="Ordered subject IDs assigned to testing."
    )

class TrainManifest(Manifest):
    """Manifest ``train`` writes for the selected subjects and recipe."""

    manifest_type: Literal["train"] = Field(
        default="train",
        description="Stable discriminator for a training run manifest.",
    )
    dataset_dir: str = Field(
        description="Absolute path to the preprocessed dataset used for training."
    )
    split_manifest_fingerprint: str = Field(
        description="Fingerprint of the split manifest used for training."
    )
    device: str = Field(description="Torch device requested for the training run.")
    seed: int | None = Field(
        default=None, ge=0, description="Random seed used for the training run."
    )
    detector_candidate_threshold: float = Field(
        ge=0,
        le=1,
        description="Detector probability threshold for student candidates.",
    )
    detector_augmentation_factor: int = Field(
        gt=0, description="Number of preprocessed variants used for detector training."
    )
    discriminator_augmentation_factor: int = Field(
        gt=0,
        description="Number of preprocessed variants used for discriminator training.",
    )
    validation_augmentation_factor: int = Field(
        gt=0, description="Number of preprocessed variants used for validation."
    )
    detector_patch_size: int = Field(
        gt=0, description="Cubic detector patch edge length in voxels."
    )
    discriminator_patch_size: int = Field(
        gt=0, description="Cubic discriminator patch edge length in voxels."
    )
    num_workers: int = Field(
        ge=0, description="Number of worker processes used by training loaders."
    )
    pin_memory: bool = Field(
        description="Whether training loaders pin batches in host memory."
    )


class TrainStageManifest(Manifest):
    """Manifest published when an individual training stage completes."""

    manifest_type: Literal["train_stage"] = Field(
        default="train_stage",
        description="Stable discriminator for a training stage manifest.",
    )
    stage: StageName = Field(
        description="Training stage represented by this manifest."
    )
    hyperparameters: Hyperparameters = Field(
        description="Optimizer and training-loop settings for this stage."
    )
    history: list[EpochLoss] = Field(
        description="Epoch loss history for the completed training stage."
    )


class InferredSubject(FrozenModel):
    """Published result metadata for one inferred preprocessed subject."""

    subject_id: str = Field(description="Subject identifier.")
    output_path: str = Field(description="Absolute path to the final detection mask.")


class InferManifest(Manifest):
    """Manifest written after the internal inference pipe completes."""

    manifest_type: Literal["inference"] = Field(
        default="inference",
        description="Stable discriminator for an inference manifest.",
    )
    device: str = Field(description="Torch device used for inference.")
    detector_checkpoint_path: str = Field(description="Detector checkpoint used.")
    student_checkpoint_path: str = Field(description="Student checkpoint used.")
    detector_threshold: float = Field(
        ge=0,
        le=1,
        description="Detector probability threshold used for candidates.",
    )
    student_threshold: float = Field(
        ge=0,
        le=1,
        description="Student probability threshold used for retained candidates.",
    )
    discriminator_patch_size: int = Field(
        gt=0,
        description="Cubic candidate patch edge length in voxels.",
    )
    minimum_volume_mm3: float = Field(
        ge=0,
        description="Minimum retained component volume in cubic millimetres.",
    )
    maximum_ellipticity: float = Field(
        ge=0,
        description="Maximum retained component ellipticity.",
    )
    minimum_brain_distance_mm: float = Field(
        ge=0,
        description="Minimum retained centroid distance from the brain boundary.",
    )
    subjects: list[InferredSubject] = Field(
        description="Results published for each inferred subject."
    )

class EvaluatedSubject(FrozenModel):
    """Evaluation metrics published for one subject."""

    subject_id: str = Field(description="Subject identifier.")
    metrics: EvaluationMetrics = Field(description="Metrics for the subject.")
class EvaluateManifest(Manifest):
    """Manifest written after scoring held-out inference results."""

    manifest_type: Literal["evaluation"] = Field(
        default="evaluation",
        description="Stable discriminator for an evaluation manifest.",
    )
    dataset_dir: str = Field(
        description="Absolute path to the preprocessed dataset evaluated."
    )
    device: str = Field(description="Torch device used for inference.")
    inference_manifest_path: str = Field(
        description="Absolute path to the inference manifest that was scored."
    )
    subjects: list[EvaluatedSubject] = Field(
        description="Metrics produced for each evaluated subject."
    )
    aggregate: EvaluationAggregate = Field(
        description="Metrics aggregated across evaluated subjects."
    )


class PatchManifest(Manifest):
    """Manifest ``patch`` writes for one materialized stage and split."""

    manifest_type: Literal["patch"] = Field(
        default="patch",
        description="Stable discriminator for a materialized patch manifest.",
    )
    stage: StageName = Field(description="Training stage owning these patches.")
    split: SplitName = Field(description="Dataset split owning these patches.")
    subject_ids: list[str] = Field(
        description="Subject IDs requested for patch extraction."
    )
    patch_size: int = Field(
        gt=0, description="Cubic patch edge length in voxels."
    )
    augmentation_factor: int = Field(
        gt=0, description="Number of preprocessed variants considered per subject."
    )
    probability_threshold: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Detector probability threshold used for target-centered patches.",
    )
    records: list[PatchRecord] = Field(
        description="Materialized patch records produced by extraction."
    )


def reject_duplicate_ids(ids, entity_name: str) -> None:
    seen: set[str] = set()
    for entity_id in ids:
        if entity_id in seen:
            raise ValueError(f"duplicate {entity_name} ID in manifest: {entity_id!r}")
        seen.add(entity_id)


