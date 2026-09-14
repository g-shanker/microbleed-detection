"""Atomic JSON, NIfTI, array, and checkpoint I/O used by core operations."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from . import utils
from .datamodels import CheckpointState


def write_json_atomic(path: Path, data: dict[str, Any] | list[Any]) -> None:
    """Serialize ``data`` to ``path`` as JSON, replacing it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(data, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    """Read and parse a JSON document from ``path``."""
    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def load_volume(path: Path | str) -> nib.Nifti1Image:
    return cast(nib.Nifti1Image, nib.load(path))


def save_volume(volume: nib.Nifti1Image, path: Path) -> None:
    nib.save(volume, path)


def nifti_to_numpy(volume: nib.Nifti1Image) -> np.ndarray:
    return volume.get_fdata()


def numpy_to_nifti(array: np.ndarray, reference: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.Nifti1Image(array, reference.affine, reference.header)


def load_array_mmap(path: str | Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def load_model_weights(
    model: nn.Module, checkpoint_path: Path
) -> None:
    target_model = utils.unwrap_model(model)
    device = utils.get_model_device(target_model)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    target_model.load_state_dict(checkpoint["model_state_dict"])


def save_array_atomic(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npy", delete=False
    ) as file:
        temporary_path = Path(file.name)
    try:
        np.save(temporary_path, array)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_checkpoint_atomic(state: CheckpointState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)
