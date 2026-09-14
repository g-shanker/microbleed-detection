import json
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pytest
import torch
import torch.nn as nn

from microbleednet.core import io


def test_numpy_to_nifti_preserves_reference_geometry() -> None:
    reference = nib.Nifti1Image(np.zeros((2, 2, 2)), np.diag([2, 3, 4, 1]))
    array = np.ones((2, 2, 2))

    volume = io.numpy_to_nifti(array, reference)

    np.testing.assert_array_equal(volume.affine, reference.affine)
    assert volume.header.get_data_shape() == array.shape


def test_save_and_load_volume_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "volume.nii.gz"
    volume = nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4))

    io.save_volume(volume, path)
    loaded = io.load_volume(path)

    np.testing.assert_array_equal(io.nifti_to_numpy(loaded), np.ones((2, 2, 2)))


def test_save_volume_preserves_geometry_and_binary_dtype(tmp_path: Path) -> None:
    reference = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4))
    path = tmp_path / "detections.nii.gz"

    io.save_volume(
        io.numpy_to_nifti(np.ones((2, 2, 2), dtype=np.uint8), reference), path
    )

    saved = cast(nib.Nifti1Image, nib.load(path))
    assert saved.shape == reference.shape
    assert np.array_equal(
        cast(np.ndarray, saved.affine), cast(np.ndarray, reference.affine)
    )
    assert saved.get_data_dtype() == np.dtype(np.uint8)
    assert set(np.unique(saved.get_fdata())) == {1.0}


def test_save_array_atomic_cleans_temporary_file_on_failure(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(io.np, "save", lambda *args: (_ for _ in ()).throw(OSError()))

    with pytest.raises(OSError):
        io.save_array_atomic(np.ones(1), tmp_path / "array.npy")

    assert list(tmp_path.iterdir()) == []


def test_load_model_weights_restores_checkpoint_state(tmp_path: Path) -> None:
    source = nn.Linear(1, 1)
    target = nn.Linear(1, 1)
    checkpoint_path = tmp_path / "model.pth"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint_path)

    io.load_model_weights(target, checkpoint_path)

    torch.testing.assert_close(target.weight, source.weight)
    torch.testing.assert_close(target.bias, source.bias)


def test_write_json_atomic_creates_parent_and_stable_json(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "data.json"

    io.write_json_atomic(path, {"zebra": 1, "apple": "microbleed"})

    content = path.read_text(encoding="utf-8")
    assert content == '{\n  "apple": "microbleed",\n  "zebra": 1\n}\n'


def test_write_json_atomic_replaces_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text("old content", encoding="utf-8")

    io.write_json_atomic(path, ["new", "content"])

    assert io.read_json(path) == ["new", "content"]
    assert list(tmp_path.iterdir()) == [path]


def test_write_json_atomic_removes_temporary_file_on_serialization_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "manifest.json"

    with pytest.raises(TypeError):
        io.write_json_atomic(path, {"invalid": object()})

    assert list(path.parent.glob("tmp*")) == []
    assert not path.exists()


def test_write_json_atomic_handles_temporary_file_creation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"

    def raise_creation_error(*args: object, **kwargs: object) -> None:
        raise OSError("could not create temporary file")

    monkeypatch.setattr(io.tempfile, "NamedTemporaryFile", raise_creation_error)

    with pytest.raises(OSError, match="could not create temporary file"):
        io.write_json_atomic(path, {"key": "value"})

    assert not path.exists()


def test_read_json_rejects_malformed_document(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text("{not valid json}", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        io.read_json(path)