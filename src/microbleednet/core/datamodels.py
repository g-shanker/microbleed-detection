from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
Shape3D = tuple[int, int, int]
BoundingBox = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
Modality = Literal["T2*-GRE", "SWI", "QSM"]
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


@dataclass(frozen=True)
class PreprocessResult:
    image: FloatArray
    mask: IntArray
    affine: FloatArray

    def __post_init__(self) -> None:
        if self.image.ndim != 3:
            raise ValueError("image must be a 3D array")
        if not np.isfinite(self.image).all():
            raise ValueError("image must contain only finite values")
        if self.mask.shape != self.image.shape:
            raise ValueError("mask must match image shape")


@dataclass(frozen=True)
class ExtractedPatches:
    volumes: np.ndarray
    masks: np.ndarray
    frst: np.ndarray


@dataclass(frozen=True)
class LoadedPatch:
    volume: FloatArray
    mask: IntArray
    has_microbleed: bool
    augmented: bool


@dataclass(frozen=True)
class PatchRecord:
    volume_path: str
    mask_path: str
    frst_path: str
    patch_index: int
    has_microbleed: bool
    augmented: bool
