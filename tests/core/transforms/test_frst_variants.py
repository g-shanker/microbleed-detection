import numpy as np
import pytest

from microbleednet.core.transforms import frst, frst_variants
from microbleednet.errors import ApplicationError

SPACING = (0.5, 0.5, 2.0)


def _volume() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    volume = rng.normal(0.4, 0.02, (40, 36, 10)).astype(np.float32)
    brain = np.zeros(volume.shape, dtype=np.uint8)
    brain[3:37, 3:33, 1:9] = 1
    volume[brain == 0] = 0.0
    volume[18:21, 18:21, 4:6] = 0.9
    return volume, brain


def test_baseline_reproduces_pipeline_frst() -> None:
    volume, brain = _volume()

    np.testing.assert_array_equal(
        frst_variants.compute(volume, brain, SPACING), frst.apply(volume, brain)
    )


@pytest.mark.parametrize(
    "names",
    [
        [name] for name in frst_variants.VARIANT_OVERRIDES
    ]
    + [
        ["3d_mm", "normalization_fixed"],
        ["radii_small", "normalization_fixed"],
        ["3d_mm", "presmooth_0.5mm", "contrast_norm"],
    ],
)
def test_every_variant_produces_a_finite_map(names: list[str]) -> None:
    volume, brain = _volume()

    result = frst_variants.compute(
        volume, brain, SPACING, frst_variants.settings_for(names)
    )

    assert result.shape == volume.shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_settings_for_merges_and_rejects_conflicts() -> None:
    merged = frst_variants.settings_for(["alpha_1", "sigma_0.25"])
    assert (merged.alpha, merged.sigma_factor) == (1.0, 0.25)

    with pytest.raises(ApplicationError, match="same option differently"):
        frst_variants.settings_for(["3d_mm", "radii_small"])
    with pytest.raises(ApplicationError, match="Unknown FRST variant"):
        frst_variants.settings_for(["missing"])


def test_volume_without_gradients_gives_zero_map() -> None:
    volume = np.ones((12, 12, 4), dtype=np.float32)

    result = frst_variants.compute(volume, np.ones_like(volume), SPACING)

    assert not result.any()


def _two_blobs(strong: float) -> tuple[np.ndarray, np.ndarray]:
    volume = np.full((48, 24, 4), 0.4, dtype=np.float32)
    volume[8:11, 10:13, :] = 0.9
    volume[36:39, 10:13, :] = strong
    return volume, np.ones_like(volume)


def test_fixed_normalization_ignores_stronger_structures_elsewhere() -> None:
    fixed = frst_variants.settings_for(["normalization_fixed", "sigma_0.1"])
    scaled = frst_variants.settings_for(["sigma_0.1"])
    alone, brain = _two_blobs(0.9)
    crowded, _ = _two_blobs(5.0)

    def response(volume: np.ndarray, settings) -> float:
        return float(frst_variants.compute(volume, brain, SPACING, settings)[9, 11, 2])

    assert response(alone, fixed) == pytest.approx(response(crowded, fixed))
    assert response(crowded, scaled) < response(alone, scaled)


def test_end_on_suppression_removes_long_structures_only() -> None:
    volume = np.full((24, 24, 12), 0.4, dtype=np.float32)
    volume[5:8, 5:8, :] = 0.9
    volume[15:18, 15:18, 5:7] = 0.9
    brain = np.ones_like(volume)
    settings = frst_variants.settings_for(["end_on_5mm"])

    result = frst_variants.compute(volume, brain, (0.5, 0.5, 1.0), settings)

    assert result[6, 6, 6] < 0.01 * result[16, 16, 5]
    assert result[16, 16, 5] > 0


def test_3d_voting_in_millimetres_peaks_at_anisotropic_sphere_center() -> None:
    spacing = np.array([0.5, 0.5, 1.5])
    grid = np.indices((40, 40, 14)).transpose(1, 2, 3, 0) * spacing
    center = np.array([20, 20, 7]) * spacing
    volume = np.where(
        np.linalg.norm(grid - center, axis=-1) <= 2.0, 0.9, 0.4
    ).astype(np.float32)
    settings = frst_variants.settings_for(["3d_mm"])

    result = frst_variants.compute(
        volume, np.ones_like(volume), tuple(spacing), settings
    )

    peak = np.unravel_index(int(np.argmax(result)), result.shape)
    assert np.abs(np.array(peak) - [20, 20, 7]).max() <= 1


def test_zscore_output_centres_brain_values() -> None:
    volume, brain = _volume()
    settings = frst_variants.settings_for(["zscore_output"])

    result = frst_variants.compute(volume, brain, SPACING, settings)

    assert result[brain > 0].mean() == pytest.approx(0.0, abs=1e-4)
    assert result[brain > 0].std() == pytest.approx(1.0, abs=1e-3)
