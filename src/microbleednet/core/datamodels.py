from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import nibabel as nib
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from ..errors import ApplicationError

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
Shape3D = tuple[int, int, int]
BoundingBox = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
VolumeArray = FloatArray
MaskArray = IntArray
VoxelSpacing = tuple[float, float, float]
Modality = Literal["T2*-GRE", "SWI", "QSM"]
INVERTED_MODALITIES: frozenset[Modality] = frozenset({"T2*-GRE", "SWI"})
TorchStateDict = dict[str, Any]


class FrozenModel(BaseModel):
    """Immutable, unknown-field-rejecting base for application data models."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CheckpointState(TypedDict):
    epoch: int
    model_state_dict: TorchStateDict
    optimizer_state_dict: TorchStateDict
    scheduler_state_dict: TorchStateDict
    scaler_state_dict: TorchStateDict
    best_val_loss: float
    epochs_without_improvement: int
    history: list["EpochLoss"]


class EpochLoss(TypedDict):
    """Mean training and validation losses completed during one epoch."""

    epoch: int
    training_loss: float
    validation_loss: float


class PatchSizes:
    """Fixed patch dimensions shared by training and inference."""

    DETECTOR = 48
    DISCRIMINATOR = 24


class Hyperparameters(FrozenModel):
    """Optimization and training-loop settings for a model run."""

    batch_size: int = Field(
        gt=0,
        multiple_of=2,
        description=(
            "Positive even number of samples per training batch shared by all "
            "models."
        ),
    )
    max_epochs: int = Field(
        ge=0, description="Maximum number of epochs permitted for all models."
    )
    patience: int = Field(
        ge=0,
        description="Epochs without validation improvement allowed for all models.",
    )
    learning_rate: float = Field(
        gt=0, description="Initial learning rate shared by all model optimizers."
    )
    adam_epsilon: float = Field(
        gt=0, description="Adam numerical-stability epsilon shared by all models."
    )
    learning_rate_factor: float = Field(
        gt=0, description="Shared factor applied when the learning rate decays."
    )
    learning_rate_period: int = Field(
        gt=0, description="Shared number of epochs between learning-rate decays."
    )
    minimum_learning_rate: float = Field(
        gt=0, description="Shared lower bound for the scheduled learning rate."
    )
    weight_decay: float = Field(
        ge=0, description="Shared L2 weight-decay coefficient for all optimizers."
    )
    minimum_improvement: float = Field(
        ge=0,
        description="Shared minimum validation-loss improvement considered meaningful.",
    )
    use_amp: bool = Field(
        description="Whether automatic mixed precision is enabled for all models."
    )


class EvaluationMetrics(FrozenModel):
    """Lesion-level metrics for one prediction/reference mask pair."""

    true_positive: int = Field(
        ge=0, description="Reference components matched by predicted components."
    )
    false_positive: int = Field(
        ge=0, description="Predicted components that did not match a reference."
    )
    false_negative: int = Field(
        ge=0, description="Reference components missed by the prediction."
    )
    cluster_tpr: float = Field(
        ge=0,
        le=1,
        description="Matched reference components divided by all reference components.",
    )
    cluster_precision: float = Field(
        ge=0,
        le=1,
        description="Matched predicted components divided by all predicted components.",
    )

class EvaluationAggregate(EvaluationMetrics):
    """Aggregated lesion-level metrics for an evaluated cohort."""

    false_positives_per_subject: float = Field(
        ge=0,
        description="Average number of unmatched predicted components per subject.",
    )


@dataclass(frozen=True)
class PreprocessInput:
    volume: nib.Nifti1Image
    mask: nib.Nifti1Image | None
    modality: Modality

    def __post_init__(self) -> None:
        if len(self.volume.shape) != 3:
            raise ApplicationError(
                category="Input data",
                summary="Preprocessing requires a 3D volume",
                cause=f"The volume has {len(self.volume.shape)} dimensions",
                fix="Provide a three-dimensional NIfTI volume",
            )
        if self.mask is not None and self.volume.shape != self.mask.shape:
            raise ApplicationError(
                category="Input data",
                summary="Volume and mask shapes do not match",
                cause=(
                    f"Volume shape is {self.volume.shape}, "
                    f"mask shape is {self.mask.shape}"
                ),
                fix="Provide a mask on the same voxel grid as the volume",
            )

        if self.volume.affine is None:
            raise ApplicationError(
                category="Input data",
                summary="Volume affine is missing",
                fix="Repair the NIfTI spatial metadata before preprocessing",
            )
        if not np.isfinite(np.asarray(self.volume.affine)).all():
            raise ApplicationError(
                category="Input data",
                summary="Volume affine is not finite",
                fix="Repair the NIfTI affine before preprocessing",
            )
        if self.mask is not None:
            if self.mask.affine is None:
                raise ApplicationError(
                    category="Input data",
                    summary="Mask affine is missing",
                    fix="Repair the mask NIfTI spatial metadata",
                )
            if not np.isfinite(np.asarray(self.mask.affine)).all():
                raise ApplicationError(
                    category="Input data",
                    summary="Mask affine is not finite",
                    fix="Repair the mask NIfTI affine before preprocessing",
                )
            if not np.allclose(self.volume.affine, self.mask.affine):
                raise ApplicationError(
                    category="Input data",
                    summary="Volume and mask affines do not match",
                    fix="Register or resample the mask to the volume grid",
                )

        volume_spacing = np.asarray(self.volume.header.get_zooms()[:3], dtype=float)
        if (
            volume_spacing.shape != (3,)
            or not np.isfinite(volume_spacing).all()
            or np.any(volume_spacing <= 0)
        ):
            raise ApplicationError(
                category="Input data",
                summary="Volume voxel spacing is invalid",
                cause=f"Observed spacing {tuple(volume_spacing)}",
                fix="Repair the NIfTI header with finite positive voxel spacing",
            )
        if self.mask is not None:
            mask_spacing = np.asarray(self.mask.header.get_zooms()[:3], dtype=float)
            if (
                mask_spacing.shape != (3,)
                or not np.isfinite(mask_spacing).all()
                or np.any(mask_spacing <= 0)
            ):
                raise ApplicationError(
                    category="Input data",
                    summary="Mask voxel spacing is invalid",
                    cause=f"Observed spacing {tuple(mask_spacing)}",
                    fix="Repair the mask NIfTI header with finite positive spacing",
                )

        volume_data = np.asarray(self.volume.get_fdata())
        if not np.isfinite(volume_data).all():
            raise ApplicationError(
                category="Input data",
                summary="Volume contains non-finite values",
                fix="Remove NaN and infinite values from the source volume",
            )
        if not np.any(volume_data > 0):
            raise ApplicationError(
                category="Input data",
                summary="Volume contains no positive voxels",
                fix="Verify the source volume and preprocessing normalization",
            )
        if self.mask is not None:
            mask_data = np.asarray(self.mask.get_fdata())
            if not np.isfinite(mask_data).all():
                raise ApplicationError(
                    category="Input data",
                    summary="Mask contains non-finite values",
                    fix="Remove NaN and infinite values from the reference mask",
                )
            if not np.isin(mask_data, [0, 1]).all():
                raise ApplicationError(
                    category="Input data",
                    summary="Mask is not binary",
                    fix="Convert the reference mask to values 0 and 1",
                )


@dataclass(frozen=True)
class PreprocessOutput:
    volume: FloatArray
    mask: IntArray | None
    affine: FloatArray
    bounding_box: BoundingBox
    original_volume: nib.Nifti1Image


@dataclass(frozen=True)
class ExtractedPatches:
    volumes: np.ndarray
    masks: np.ndarray
    frst: np.ndarray


@dataclass(frozen=True)
class LoadedPatch:
    volume: VolumeArray
    mask: MaskArray
    has_microbleed: bool


class PatchRecord(FrozenModel):
    """One materialized patch array and its training label."""

    volume_path: str = Field(description="Absolute path to the patch volumes.")
    mask_path: str = Field(description="Absolute path to the patch masks.")
    frst_path: str = Field(description="Absolute path to the patch FRST arrays.")
    patch_index: int = Field(ge=0, description="Index within the source patch array.")
    has_microbleed: bool = Field(
        description="Whether the patch mask contains a microbleed."
    )

