import numpy as np
import pytest

from microbleednet.core.transforms import augmentations
from tests.support_core import voxels


class ControlledGenerator:
    def __init__(self, transformation: str):
        self.transformation = transformation
        self.integer_values = iter((1, 1, -1))

    def integers(self, low, high):
        return next(self.integer_values)

    def choice(self, transformations, size, replace):
        return np.array([self.transformation])

    def uniform(self, low, high):
        return (low + high) / 2

    def normal(self, mean, standard_deviation, shape):
        return np.full(shape, mean + standard_deviation)


@pytest.fixture
def force_transformation(monkeypatch):
    def force(transformation: str) -> None:
        monkeypatch.setattr(
            augmentations.np.random,
            "default_rng",
            lambda: ControlledGenerator(transformation),
        )

    return force


def test_augment_preserves_shape_and_binary_mask() -> None:
    center = (16, 16, 2)
    volume = voxels((32, 32, 4), center)
    mask = voxels((32, 32, 4), center, dtype=np.uint8)

    transformed_volume, transformed_mask = augmentations.augment(
        volume, mask, np.random.default_rng(0)
    )

    assert transformed_mask is not None
    assert transformed_volume.shape == volume.shape
    assert transformed_mask.shape == mask.shape
    assert set(np.unique(transformed_mask)).issubset({0, 1})


def test_augment_is_reproducible_with_matching_generators() -> None:
    volume = np.arange(64, dtype=float).reshape((4, 4, 4))

    first, _ = augmentations.augment(volume, None, np.random.default_rng(42))
    second, _ = augmentations.augment(volume, None, np.random.default_rng(42))

    np.testing.assert_array_equal(first, second)


def test_translate_applies_same_random_offsets_to_volume_and_mask(
    force_transformation,
) -> None:
    volume = voxels((32, 32, 1), (16, 16, 0))
    mask = voxels((32, 32, 1), (16, 16, 0), dtype=np.uint8)
    force_transformation("translate")

    translated_volume, translated_mask = augmentations.augment(volume, mask)

    assert translated_mask is not None
    assert translated_volume.sum() == 1
    assert not translated_volume[16, 16, 0]
    np.testing.assert_array_equal(translated_volume > 0, translated_mask > 0)


def test_translate_supports_missing_mask(force_transformation) -> None:
    force_transformation("translate")

    translated_volume, translated_mask = augmentations.augment(
        voxels((32, 32, 1), (16, 16, 0)), None
    )

    assert translated_volume.sum() == 1
    assert translated_mask is None


@pytest.mark.parametrize("transformation", ["noise", "blur"])
def test_intensity_augmentations_change_volume_but_not_mask(
    force_transformation, transformation: str
) -> None:
    volume = voxels((3, 3, 3), (1, 1, 1))
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    force_transformation(transformation)

    transformed_volume, transformed_mask = augmentations.augment(volume, mask)

    assert transformed_mask is mask
    assert transformed_volume.shape == volume.shape
    assert not np.array_equal(transformed_volume, volume)
