import numpy as np
import torch

from microbleednet.core.transforms import frst


def test_normalize_tensor_slicewise_handles_variable_and_constant_slices() -> None:
    tensor = torch.tensor([[[1.0, 3.0]], [[2.0, 2.0]]])

    result = frst.normalize_tensor_slicewise(tensor)

    torch.testing.assert_close(
        result,
        torch.tensor([[[0.0, 1.0]], [[0.0, 0.0]]]),
    )


def test_apply_processes_non_degenerate_volume() -> None:
    result = frst.apply(np.ones((2, 2, 2)))

    assert result.shape == (2, 2, 2)
