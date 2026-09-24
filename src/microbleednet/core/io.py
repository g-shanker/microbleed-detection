"""Atomic JSON, NIfTI, array, and checkpoint I/O used by core operations."""

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from ..errors import ApplicationError
from . import utils
from .datamodels import CheckpointState


@contextmanager
def atomic_path(path: Path, suffix: str) -> Generator[Path, None, None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=suffix, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        yield temporary_path
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_json(path: Path, data: dict[str, Any] | list[Any]) -> None:
    """Serialize ``data`` to ``path`` as JSON, replacing it atomically."""
    with atomic_path(path, suffix="") as temporary_path:
        with temporary_path.open("w", encoding="utf-8") as temporary_file:
            json.dump(data, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())


def read_json(path: Path) -> Any:
    """Read and parse a JSON document from ``path``."""
    with path.open(encoding="utf-8") as json_file:
        payload = json.load(json_file)
    return payload


def load_volume(path: Path | str) -> nib.Nifti1Image:
    volume = nib.load(path)
    if not isinstance(volume, nib.Nifti1Image):
        raise ApplicationError(
            category="Input data",
            summary="Unsupported NIfTI image type",
            cause=f"'{path}' is not a NIfTI-1 image",
            fix="Convert the input to the supported NIfTI-1 format",
            context={"path": str(path)},
        )
    return volume


def save_volume(volume: nib.Nifti1Image, path: Path) -> None:
    with atomic_path(path, suffix="".join(path.suffixes)) as temporary_path:
        nib.save(volume, temporary_path)


def nifti_to_numpy(volume: nib.Nifti1Image) -> np.ndarray:
    return volume.get_fdata()


def numpy_to_nifti(array: np.ndarray, reference: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.Nifti1Image(array, reference.affine, reference.header)


def load_array_mmap(path: str | Path) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    return array


def load_model_weights(
    model: nn.Module, checkpoint_path: Path
) -> None:
    target_model = utils.unwrap_model(model)
    device = utils.get_model_device(target_model)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    target_model.load_state_dict(checkpoint["model_state_dict"])


def save_array(array: np.ndarray, path: Path) -> None:
    with atomic_path(path, suffix=".npy") as temporary_path:
        np.save(temporary_path, array)


def save_checkpoint(state: CheckpointState, path: Path) -> None:
    with atomic_path(path, suffix=path.suffix + ".tmp") as temporary_path:
        with temporary_path.open("wb") as temporary_file:
            torch.save(state, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
