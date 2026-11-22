from typing import Literal, cast

import numpy as np
import pytest
from pydantic import ValidationError

from experiments.frst import variants
from experiments.frst.variants import BASELINE, FrstSpec
from microbleednet.core.transforms import frst


def disc_stack(
    shape: tuple[int, int, int] = (25, 25, 5), radius: float = 4, slices=slice(None)
) -> np.ndarray:
    """A bright disc (or cylinder) centred in the middle of a dark volume."""
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    center = (shape[0] // 2, shape[1] // 2)
    disc = ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2).astype(np.float32)
    volume = np.zeros(shape, dtype=np.float32)
    volume[:, :, slices] = disc[:, :, None]
    return volume


def full_mask(volume: np.ndarray) -> np.ndarray:
    return np.ones(volume.shape, dtype=bool)


def test_default_spec_reproduces_the_pipeline_frst() -> None:
    volume = np.random.default_rng(0).random((30, 30, 4)).astype(np.float32)

    actual = variants.apply(volume, full_mask(volume), (1, 1, 1), BASELINE)

    np.testing.assert_allclose(actual, frst.apply(volume), atol=1e-5)


@pytest.mark.parametrize("mode", ["slice", "volume"])
def test_response_peaks_at_the_centre_of_a_bright_disc(
    mode: Literal["slice", "volume"],
) -> None:
    volume = disc_stack()
    spec = FrstSpec(mode=mode, radii=(4,))

    result = variants.apply(volume, full_mask(volume), (1, 1, 1), spec)

    middle = result[:, :, 2]
    assert np.unravel_index(middle.argmax(), middle.shape) == (12, 12)


def test_constant_volume_has_no_voters_and_a_zero_response() -> None:
    volume = np.ones((8, 8, 3), dtype=np.float32)

    result = variants.apply(volume, full_mask(volume), (1, 1, 1), BASELINE)

    assert result.shape == volume.shape
    assert not result.any()


@pytest.mark.parametrize(
    "overrides",
    [{"radii": ()}, {"gradient_quantile": 1.0}, {"alpha": 0}, {"unknown": 1}],
)
def test_spec_rejects_invalid_values(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        FrstSpec(**overrides)


def test_gradient_quantile_removes_the_weakest_voters() -> None:
    volume = np.random.default_rng(1).random((16, 16, 3)).astype(np.float32)
    gradients = cast(list[np.ndarray], np.gradient(volume, axis=(0, 1)))
    magnitude = np.sqrt(gradients[0] ** 2 + gradients[1] ** 2)
    mask = full_mask(volume)

    everyone = variants._voters(magnitude, mask, BASELINE)
    strong = variants._voters(magnitude, mask, FrstSpec(gradient_quantile=0.5))

    assert 0 < strong.sum() < everyone.sum()
    assert magnitude[strong].min() >= magnitude[everyone & ~strong].max()


def test_edge_margin_excludes_voters_next_to_the_brain_boundary() -> None:
    magnitude = np.ones((9, 9, 1))
    mask = np.zeros((9, 9, 1), dtype=bool)
    mask[1:8, 1:8] = True

    voters = variants._voters(magnitude, mask, FrstSpec(edge_margin=2))

    expected = np.zeros_like(mask)
    expected[3:6, 3:6] = True
    np.testing.assert_array_equal(voters, expected)


def test_slice_normalization_scales_every_slice_to_unit_range() -> None:
    ramp = np.arange(4.0).reshape(2, 2)
    tensor = np.stack([ramp, 10 * ramp], axis=2)

    result = variants._normalize(tensor, "slice")

    np.testing.assert_allclose(result.min(axis=(0, 1)), [0, 0])
    np.testing.assert_allclose(result.max(axis=(0, 1)), [1, 1])


def test_volume_normalization_divides_by_the_global_maximum() -> None:
    tensor = np.array([[[2.0, 4.0]]])

    np.testing.assert_allclose(variants._normalize(tensor, "volume"), [[[0.5, 1.0]]])


def test_through_slice_blur_spreads_the_response_to_neighbouring_slices() -> None:
    volume = disc_stack(slices=slice(2, 3))
    mask = full_mask(volume)

    sharp = variants.apply(volume, mask, (1, 1, 1), BASELINE)
    blurred = variants.apply(
        volume, mask, (1, 1, 1), FrstSpec(through_slice_blur_mm=1.0)
    )

    assert not sharp[:, :, 1].any()
    assert blurred[:, :, 1].max() > 0
