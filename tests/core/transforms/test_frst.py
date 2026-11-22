import numpy as np
import torch

from microbleednet.core.transforms import frst
from tests.support_core import voxels


def test_normalize_tensor_slicewise_handles_variable_and_constant_slices() -> None:
    tensor = torch.tensor([[[1.0, 3.0]], [[2.0, 2.0]]])

    result = frst.normalize_tensor_slicewise(tensor)

    torch.testing.assert_close(result, torch.tensor([[[0.0, 1.0]], [[0.0, 0.0]]]))


def test_apply_processes_non_degenerate_volume() -> None:
    volume = voxels((9, 9, 3), (slice(2, 7), slice(2, 7)), dtype=np.float32)
    volume[4, 4, :] = 2.0

    result = frst.apply(volume)

    assert result.shape == (9, 9, 3)
    assert np.isfinite(result).all()
    assert np.max(result) > 0
