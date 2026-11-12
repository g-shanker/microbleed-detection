import numpy as np
import pytest
import torch

from microbleednet.core.transforms import frst


def test_normalize_tensor_volume_uses_one_scale_for_all_slices() -> None:
    tensor = torch.tensor([[[1.0, 2.0]], [[4.0, 0.0]]])

    result = frst.normalize_tensor_volume(tensor)

    torch.testing.assert_close(
        result, torch.tensor([[[0.25, 0.5]], [[1.0, 0.0]]])
    )


def test_normalize_tensor_volume_leaves_empty_tensor_unchanged() -> None:
    tensor = torch.zeros(2, 3, 3)

    assert frst.normalize_tensor_volume(tensor) is tensor


def _structured_volume() -> np.ndarray:
    volume = np.zeros((9, 9, 3), dtype=np.float32)
    volume[2:7, 2:7, :] = 1.0
    volume[4, 4, :] = 2.0
    return volume


def test_apply_processes_non_degenerate_volume() -> None:
    volume = _structured_volume()

    result = frst.apply(volume, np.ones_like(volume))

    assert result.shape == (9, 9, 3)
    assert np.isfinite(result).all()
    assert np.max(result) > 0


def test_apply_ignores_brain_boundary_gradients() -> None:
    volume = np.zeros((25, 25, 2), dtype=np.float32)
    volume[4:21, 4:21] = 1.0

    result = frst.apply(volume, volume > 0)

    assert np.all(result == 0)


def test_apply_response_does_not_depend_on_other_slice_content() -> None:
    volume = np.ones((25, 25, 2), dtype=np.float32)
    volume[5:8, 5:8, :] = 2.0
    volume[16:19, 16:19, 1] = 5.0

    result = frst.apply(volume, np.ones_like(volume))

    assert result[6, 6, 0] > 0
    assert result[6, 6, 0] == pytest.approx(result[6, 6, 1])
