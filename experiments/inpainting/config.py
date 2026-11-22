"""Experiment configuration (see ``config.example.toml``)."""

import tomllib
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from microbleednet.core.datamodels import BiasFieldCorrection

from .detectors import Strategy


class InpaintingConfig(BaseModel):
    """Settings of one dilation experiment run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_dir: Path = Field(description="Indexed raw dataset with lesion masks.")
    bias_field_correction: BiasFieldCorrection = Field(
        description="Bias-field correction used for the preprocessing it runs."
    )
    output_dir: Path = Field(
        default=Path("experiments/inpainting/results"),
        description="Receives results.csv, the saved masks and the per-subject cache.",
    )
    strategies: tuple[Strategy, ...] = Field(
        default=("axial", "frangi3d", "frangi3d_range"),
        min_length=1,
        description=(
            "Vessel detection strategies to compare: 'axial' is the pipeline's "
            "slice-by-slice detector, 'frangi3d' runs the same recipe in 3D, and "
            "'frangi3d_range' does too with the Frangi scales read as "
            "(start, stop, step)."
        ),
    )
    save_masks: bool = Field(
        default=True,
        description=(
            "Write each subject's volume, lesion and brain masks and every "
            "strategy and iteration count's dilated vessel mask as NIfTI files."
        ),
    )
    save_inpainted: bool = Field(
        default=False,
        description=(
            "Also write the inpainted volume for every strategy and iteration "
            "count (large: float data). Requires save_masks."
        ),
    )
    iterations: tuple[int, ...] = Field(
        default=(0, 1, 2, 3, 4, 5, 6),
        min_length=1,
        description="In-plane dilation iteration counts to compare. 0 is no dilation.",
    )
    max_subjects: int | None = Field(
        default=None, gt=0, description="Random subset size; omit to use all subjects."
    )
    seed: int = Field(default=0, description="Seed of the subject subset.")
    lesion_margin: int = Field(
        default=1, ge=0, description="Voxels a lesion is widened by before scoring."
    )
    exclusion_radius: int = Field(
        default=4, ge=0, description="Voxels around lesions with no background peaks."
    )
    peak_min_distance: int = Field(
        default=3, ge=1, description="Minimum voxel distance between background peaks."
    )
    recall_at: tuple[int, ...] = Field(
        default=(5, 20), min_length=1, description="K values of the recall@K metric."
    )
    max_lesion_loss: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Largest share of lesion voxels a recommended setting may erase.",
    )
    auc_tolerance: float = Field(
        default=0.01,
        ge=0,
        description="A smaller setting is preferred if its mean_auc is this close.",
    )

    @field_validator("iterations")
    @classmethod
    def valid_iterations(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        """Require unique, non-negative counts."""
        if min(values) < 0 or len(set(values)) != len(values):
            raise ValueError("iterations must be unique and not negative")
        return tuple(sorted(values))

    @field_validator("strategies")
    @classmethod
    def unique_strategies(cls, values: tuple[Strategy, ...]) -> tuple[Strategy, ...]:
        """Require each strategy to be listed once."""
        if len(set(values)) != len(values):
            raise ValueError("strategies must be unique")
        return values

    @model_validator(mode="after")
    def inpainted_volumes_need_masks(self) -> "InpaintingConfig":
        """Saving inpainted volumes is part of saving masks."""
        if self.save_inpainted and not self.save_masks:
            raise ValueError("save_inpainted requires save_masks")
        return self


def load_config(path: Path) -> InpaintingConfig:
    """Read and validate an experiment TOML file."""
    with path.open("rb") as file:
        return InpaintingConfig.model_validate(tomllib.load(file))
