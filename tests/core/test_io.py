from pathlib import Path

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