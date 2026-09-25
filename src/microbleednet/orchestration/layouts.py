from pathlib import Path
from typing import Literal

from pydantic import Field

from ..constants import (
    BEST_CHECKPOINT_TEMPLATE,
    DEFAULT_NIFTI_SUFFIX,
    EVALUATION_MANIFEST_PATH,
    INFERENCE_MANIFEST_PATH,
    INFERENCE_OUTPUT_TEMPLATE,
    LATEST_CHECKPOINT_TEMPLATE,
    PATCH_DIR_TEMPLATE,
    PATCH_MANIFEST_TEMPLATE,
    PREPROCESSED_FRST_PATH,
    PREPROCESSED_MANIFEST_PATH,
    PREPROCESSED_MASKS_PATH,
    PREPROCESSED_VOLUMES_PATH,
    RAW_MANIFEST_PATH,
    SPLIT_MANIFEST_PATH,
    STAGE_MANIFEST_TEMPLATE,
    TRAIN_MANIFEST_PATH,
)
from ..core.datamodels import FrozenModel

StageName = Literal["detector", "teacher", "student"]
SplitName = Literal["train", "validation", "test"]


class DatasetLayout(FrozenModel):
    """Paths within an indexed dataset directory."""

    dataset_dir: Path = Field(
        description="Root directory for this dataset's artifacts."
    )
    raw_manifest: Path = Field(
        default=RAW_MANIFEST_PATH,
        description="Raw dataset manifest written by index-data.",
    )
    preprocessed_manifest: Path = Field(
        default=PREPROCESSED_MANIFEST_PATH,
        description=("Preprocessed dataset manifest written by preprocess."),
    )
    preprocessed_volumes_dir: Path = Field(
        default=PREPROCESSED_VOLUMES_PATH,
        description=("Directory for preprocessed volumes."),
    )
    preprocessed_masks_dir: Path = Field(
        default=PREPROCESSED_MASKS_PATH,
        description=("Directory for preprocessed masks."),
    )
    preprocessed_frst_dir: Path = Field(
        default=PREPROCESSED_FRST_PATH,
        description="Directory for precomputed FRST volumes.",
    )
    volume_suffix: str = Field(
        default=DEFAULT_NIFTI_SUFFIX,
        description="Filename suffix for a preprocessed volume.",
    )
    mask_suffix: str = Field(
        default=DEFAULT_NIFTI_SUFFIX,
        description="Filename suffix for a preprocessed mask.",
    )
    frst_suffix: str = Field(
        default=DEFAULT_NIFTI_SUFFIX,
        description="Filename suffix for a precomputed FRST volume.",
    )

    def raw_manifest_path(self) -> Path:
        """Return the indexed raw dataset manifest path."""
        return self.dataset_dir / self.raw_manifest

    def preprocessed_manifest_path(self) -> Path:
        """Return the preprocessed dataset manifest path."""
        return self.dataset_dir / self.preprocessed_manifest

    def preprocessed_volumes_path(self) -> Path:
        """Return the directory containing preprocessed volumes."""
        return self.dataset_dir / self.preprocessed_volumes_dir

    def preprocessed_masks_path(self) -> Path:
        """Return the directory containing preprocessed masks."""
        return self.dataset_dir / self.preprocessed_masks_dir

    def preprocessed_frst_path(self) -> Path:
        """Return the directory containing precomputed FRST volumes."""
        return self.dataset_dir / self.preprocessed_frst_dir

    def variant_volume_path(self, subject_id: str, variant_index: int) -> Path:
        """Return a subject variant's preprocessed volume path."""
        return self.preprocessed_volumes_path() / (
            f"{subject_id}_variant_{variant_index}{self.volume_suffix}"
        )

    def variant_mask_path(self, subject_id: str, variant_index: int) -> Path:
        """Return a subject variant's preprocessed mask path."""
        return self.preprocessed_masks_path() / (
            f"{subject_id}_variant_{variant_index}{self.mask_suffix}"
        )

    def variant_frst_path(self, subject_id: str, variant_index: int) -> Path:
        """Return a subject variant's precomputed FRST path."""
        return self.preprocessed_frst_path() / (
            f"{subject_id}_variant_{variant_index}{self.frst_suffix}"
        )


class ExperimentLayout(FrozenModel):
    experiment_dir: Path = Field(
        description="Root directory for this experiment's artifacts."
    )
    train_manifest: Path = Field(
        default=TRAIN_MANIFEST_PATH,
        description="Training split and recipe manifest.",
    )
    split_manifest: Path = Field(
        default=SPLIT_MANIFEST_PATH,
        description="Subject split manifest consumed by training.",
    )
    patch_manifest_template: Path = Field(
        default=PATCH_MANIFEST_TEMPLATE,
        description="Template for a stage and split patch manifest.",
    )
    best_checkpoint_template: Path = Field(
        default=BEST_CHECKPOINT_TEMPLATE,
        description="Relative template for the best checkpoint path.",
    )
    latest_checkpoint_template: Path = Field(
        default=LATEST_CHECKPOINT_TEMPLATE,
        description="Relative template for the latest resumable checkpoint path.",
    )
    stage_manifest_template: Path = Field(
        default=STAGE_MANIFEST_TEMPLATE,
        description="Relative template for a training stage manifest.",
    )
    patch_dir_template: Path = Field(
        default=PATCH_DIR_TEMPLATE,
        description="Relative template for a stage and split's patch directory.",
    )
    inference_manifest: Path = Field(
        default=INFERENCE_MANIFEST_PATH,
        description="Inference manifest written after processing subjects.",
    )
    evaluation_manifest: Path = Field(
        default=EVALUATION_MANIFEST_PATH,
        description="Evaluation manifest written after scoring subjects.",
    )
    inference_output_template: Path = Field(
        default=INFERENCE_OUTPUT_TEMPLATE,
        description="Template for a subject's final detection mask.",
    )

    def resolve(self, template: Path, **values: str) -> Path:
        """Resolve a formatted artifact template beneath the experiment root."""
        return self.experiment_dir / str(template).format(**values)

    def best_checkpoint_path(self, stage: StageName) -> Path:
        """Return the best checkpoint path for a training stage."""
        return self.resolve(self.best_checkpoint_template, stage=stage)

    def latest_checkpoint_path(self, stage: StageName) -> Path:
        """Return the latest resumable checkpoint path for a training stage."""
        return self.resolve(self.latest_checkpoint_template, stage=stage)

    def stage_manifest_path(self, stage: StageName) -> Path:
        """Return the lifecycle manifest path for a training stage."""
        return self.resolve(self.stage_manifest_template, stage=stage)

    def train_manifest_path(self) -> Path:
        """Return the training recipe manifest path."""
        return self.experiment_dir / self.train_manifest

    def split_manifest_path(self) -> Path:
        """Return the subject split manifest path."""
        return self.experiment_dir / self.split_manifest

    def patch_manifest_path(self, stage: StageName, split: SplitName) -> Path:
        """Return the patch manifest path for a stage and dataset split."""
        return self.resolve(self.patch_manifest_template, stage=stage, split=split)

    def patch_dir_path(self, stage: StageName, split: SplitName) -> Path:
        """Return the patch directory for a stage and dataset split."""
        return self.resolve(self.patch_dir_template, stage=stage, split=split)

    def inference_manifest_path(self) -> Path:
        """Return the inference result manifest path."""
        return self.experiment_dir / self.inference_manifest

    def evaluation_manifest_path(self) -> Path:
        """Return the evaluation result manifest path."""
        return self.experiment_dir / self.evaluation_manifest

    def inference_output_path(self, subject_id: str) -> Path:
        """Return a subject's final inference mask path."""
        return self.resolve(self.inference_output_template, subject_id=subject_id)

    def patch_volume_path(
        self, stage: StageName, split: SplitName, subject_id: str, variant: int
    ) -> Path:
        """Return a materialized patch volume path."""
        return self.patch_dir_path(stage, split) / (
            f"volumes_{subject_id}_variant_{variant}.npy"
        )

    def patch_mask_path(
        self, stage: StageName, split: SplitName, subject_id: str, variant: int
    ) -> Path:
        """Return a materialized patch mask path."""
        return self.patch_dir_path(stage, split) / (
            f"masks_{subject_id}_variant_{variant}.npy"
        )

    def patch_frst_path(
        self, stage: StageName, split: SplitName, subject_id: str, variant: int
    ) -> Path:
        """Return a materialized patch FRST path."""
        return self.patch_dir_path(stage, split) / (
            f"frst_{subject_id}_variant_{variant}.npy"
        )
