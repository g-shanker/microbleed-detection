import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest
import torch
import torch.nn as nn

from microbleednet.core import io
from microbleednet.core.datamodels import CheckpointState
from microbleednet.errors import ApplicationError
from tests.support import nifti


def _save_volume(path: Path) -> None:
    io.save_volume(nifti(data=np.arange(3.0).reshape(3, 1, 1)), path)


def _save_array(path: Path) -> None:
    io.save_array(np.arange(3), path)


def _save_checkpoint(path: Path) -> None:
    io.save_checkpoint(cast(CheckpointState, {"values": torch.arange(3)}), path)


def _load_volume(path: Path) -> list:
    return io.nifti_to_numpy(io.load_volume(path)).ravel().tolist()


def _load_array(path: Path) -> list:
    return np.load(path).tolist()


def _load_checkpoint(path: Path) -> list:
    return torch.load(path, weights_only=True)["values"].tolist()


def _raise_os_error(*args: object, **kwargs: object) -> None:
    raise OSError("write failed")


# (module, function the save delegates to, filename, save, load)
ATOMIC_SAVES: list[tuple[ModuleType, str, str, Callable, Callable]] = [
    (io.nib, "save", "volume.nii.gz", _save_volume, _load_volume),
    (io.np, "save", "array.npy", _save_array, _load_array),
    (io.torch, "save", "model.pth", _save_checkpoint, _load_checkpoint),
]
atomic_saves = pytest.mark.parametrize(
    ("backend", "attribute", "filename", "save", "load"),
    ATOMIC_SAVES,
    ids=["volume", "array", "checkpoint"],
)


@atomic_saves
def test_atomic_save_writes_destination(
    tmp_path: Path,
    backend: ModuleType,
    attribute: str,
    filename: str,
    save: Callable,
    load: Callable,
) -> None:
    path = tmp_path / "nested" / filename

    save(path)

    assert load(path) == [0, 1, 2]
    assert list(path.parent.iterdir()) == [path]


@atomic_saves
def test_atomic_save_removes_temporary_file_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: ModuleType,
    attribute: str,
    filename: str,
    save: Callable,
    load: Callable,
) -> None:
    monkeypatch.setattr(backend, attribute, _raise_os_error)
    path = tmp_path / "nested" / filename

    with pytest.raises(OSError, match="write failed"):
        save(path)

    assert list(path.parent.iterdir()) == []


def test_numpy_to_nifti_preserves_reference_geometry() -> None:
    reference = nifti(affine=np.diag([2, 3, 4, 1]))

    volume = io.numpy_to_nifti(np.ones((2, 2, 2)), reference)

    np.testing.assert_array_equal(volume.affine, reference.affine)
    assert volume.shape == (2, 2, 2)


def test_save_volume_round_trip_preserves_affine_and_dtype(tmp_path: Path) -> None:
    reference = nifti(value=0, dtype=np.uint8, affine=np.diag([2, 3, 4, 1]))
    path = tmp_path / "detections.nii.gz"

    mask = io.numpy_to_nifti(np.ones((2, 2, 2), dtype=np.uint8), reference)
    io.save_volume(mask, path)
    loaded = io.load_volume(path)

    np.testing.assert_array_equal(loaded.affine, reference.affine)
    assert loaded.get_data_dtype() == np.dtype(np.uint8)
    np.testing.assert_array_equal(io.nifti_to_numpy(loaded), np.ones((2, 2, 2)))


def test_load_volume_rejects_non_nifti_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(io.nib, "load", lambda _: object())

    with pytest.raises(ApplicationError, match="Unsupported NIfTI image type"):
        io.load_volume("volume.nii.gz")


def test_load_model_weights_restores_checkpoint_state(tmp_path: Path) -> None:
    source, target = nn.Linear(1, 1), nn.Linear(1, 1)
    checkpoint_path = tmp_path / "model.pth"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint_path)

    io.load_model_weights(target, checkpoint_path)

    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)


def test_write_json_creates_parent_and_stable_json(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "data.json"

    io.write_json(path, {"zebra": 1, "apple": "microbleed"})

    assert path.read_text(encoding="utf-8") == (
        '{\n  "apple": "microbleed",\n  "zebra": 1\n}\n'
    )


def test_write_json_replaces_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text("old content", encoding="utf-8")

    io.write_json(path, ["new", "content"])

    assert io.read_json(path) == ["new", "content"]
    assert list(tmp_path.iterdir()) == [path]


def test_write_json_removes_temporary_file_on_serialization_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "manifest.json"
    invalid: Any = {"invalid": object()}

    with pytest.raises(TypeError):
        io.write_json(path, invalid)

    assert list(path.parent.iterdir()) == []


def test_write_json_propagates_temporary_file_creation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(io.tempfile, "NamedTemporaryFile", _raise_os_error)
    path = tmp_path / "manifest.json"

    with pytest.raises(OSError, match="write failed"):
        io.write_json(path, {"key": "value"})

    assert not path.exists()


def test_read_json_rejects_malformed_document(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text("{not valid json}", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        io.read_json(path)
