import numpy as np

from microbleednet.core.transforms import augmentations


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


def test_augment_preserves_shape_and_binary_mask() -> None:
    volume = np.zeros((32, 32, 4))
    mask = np.zeros_like(volume, dtype=np.uint8)
    volume[16, 16, 2] = 1
    mask[16, 16, 2] = 1

    transformed_volume, transformed_mask = augmentations.augment(volume, mask)

    assert transformed_mask is not None
    assert transformed_volume.shape == volume.shape
    assert transformed_mask.shape == mask.shape
    assert set(np.unique(transformed_mask)).issubset({0, 1})


def test_translate_applies_same_random_offsets_to_volume_and_mask(monkeypatch) -> None:
    volume = np.zeros((32, 32, 1))
    mask = np.zeros_like(volume, dtype=np.uint8)
    volume[16, 16, 0] = 1
    mask[16, 16, 0] = 1

    monkeypatch.setattr(
        augmentations.np.random,
        "default_rng",
        lambda: ControlledGenerator("translate"),
    )
    translated_volume, translated_mask = augmentations.augment(volume, mask)
    assert translated_mask is not None

    np.testing.assert_array_equal(translated_volume > 0, translated_mask > 0)


def test_intensity_augmentations_do_not_change_mask(monkeypatch) -> None:
    volume = np.ones((3, 3, 3))
    mask = np.ones((3, 3, 3), dtype=np.uint8)

    monkeypatch.setattr(
        augmentations.np.random,
        "default_rng",
        lambda: ControlledGenerator("noise"),
    )
    noisy_volume, noisy_mask = augmentations.augment(volume, mask)
    assert noisy_mask is not None
    monkeypatch.setattr(
        augmentations.np.random,
        "default_rng",
        lambda: ControlledGenerator("blur"),
    )
    blurred_volume, blurred_mask = augmentations.augment(volume, mask)
    assert blurred_mask is not None

    assert noisy_mask is mask
    assert blurred_mask is mask
    assert not np.array_equal(noisy_volume, volume)
    assert blurred_volume.shape == volume.shape


def test_translate_supports_missing_mask(monkeypatch) -> None:
    volume = np.zeros((32, 32, 1))
    monkeypatch.setattr(
        augmentations.np.random,
        "default_rng",
        lambda: ControlledGenerator("translate"),
    )

    translated_volume, translated_mask = augmentations.augment(volume, None)

    assert translated_volume.shape == volume.shape
    assert translated_mask is None