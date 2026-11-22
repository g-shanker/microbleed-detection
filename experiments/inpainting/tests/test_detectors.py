import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from experiments.inpainting import detectors
from experiments.inpainting.tests.conftest import SerialParallel
from microbleednet.constants import (
    FRANGI_ALPHA,
    FRANGI_BETA,
    FRANGI_BLACK_RIDGES,
    FRANGI_SIGMAS,
)
from microbleednet.core.transforms import inpaint_vessels


def objects(size: int = 28) -> dict[str, np.ndarray]:
    """Boolean masks of a tube along z, a tube along x and a sphere."""
    z, y, x = np.mgrid[:size, :size, :size]
    return {
        "tube_z": (x - 7) ** 2 + (y - 7) ** 2 <= 4,
        "tube_x": ((z - 20) ** 2 + (y - 8) ** 2 <= 4) & (x > 5),
        "blob": (x - 20) ** 2 + (y - 20) ** 2 + (z - 10) ** 2 <= 9,
    }


def synthetic_volume(shapes: dict[str, np.ndarray]) -> np.ndarray:
    rng = np.random.default_rng(0)
    volume = 0.3 + rng.normal(0, 0.01, next(iter(shapes.values())).shape)
    for shape in shapes.values():
        volume[shape] = 1.0
    return gaussian_filter(volume, 0.7)


@pytest.fixture(scope="module")
def synthetic() -> tuple[dict[str, np.ndarray], np.ndarray]:
    shapes = objects()
    return shapes, detectors.frangi3d_vessel_mask(synthetic_volume(shapes))


def test_the_registry_offers_every_strategy() -> None:
    assert set(detectors.DETECTORS) == {"axial", "frangi3d", "frangi3d_range"}


def test_the_scale_range_expands_the_pipelines_start_stop_and_step() -> None:
    np.testing.assert_allclose(detectors.frangi_scale_range(), [0.5, 0.7, 0.9, 1.1])


def test_the_range_strategy_passes_the_expanded_scales_and_the_same_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    real = detectors.frangi

    def spy(image, **options):
        captured.update(options)
        return real(image, **options)

    monkeypatch.setattr(detectors, "frangi", spy)

    detectors.frangi3d_range_vessel_mask(synthetic_volume(objects(12)))

    np.testing.assert_allclose(captured.pop("sigmas"), [0.5, 0.7, 0.9, 1.1])
    assert captured == {
        "beta": FRANGI_BETA,
        "alpha": FRANGI_ALPHA,
        "black_ridges": FRANGI_BLACK_RIDGES,
    }


def test_the_range_strategy_differs_from_the_pipelines_scales_only_in_sigmas() -> None:
    shapes = objects()
    volume = synthetic_volume(shapes)

    as_pipeline = detectors.frangi3d_vessel_mask(volume)
    as_range = detectors.frangi3d_range_vessel_mask(volume)
    same_scales = detectors.frangi3d_vessel_mask(
        volume, sigmas=detectors.frangi_scale_range()
    )

    np.testing.assert_array_equal(as_range, same_scales)
    assert as_range.shape == as_pipeline.shape


def test_the_range_strategy_finds_tubes_and_not_blobs() -> None:
    shapes = objects()

    mask = detectors.frangi3d_range_vessel_mask(synthetic_volume(shapes))

    assert (mask & shapes["tube_z"]).sum() / shapes["tube_z"].sum() > 0.5
    assert (mask & shapes["tube_x"]).sum() / shapes["tube_x"].sum() > 0.5
    assert not (mask & shapes["blob"]).any()


def test_the_axial_detector_runs_the_pipelines_slice_detector_on_every_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake(image_slice: np.ndarray) -> np.ndarray:
        calls.append(image_slice.shape)
        return image_slice > 0.5

    monkeypatch.setattr(inpaint_vessels, "get_slice_vessel_mask", fake)
    volume = np.zeros((4, 5, 3))
    volume[1, 2, 2] = 1

    mask = detectors.axial_vessel_mask(volume)

    assert calls == [(4, 5)] * 3
    assert mask.sum() == 1 and mask[1, 2, 2]


def test_the_axial_detector_runs_slices_in_parallel_like_the_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = []

    class Recording(SerialParallel):
        def __init__(self, **options) -> None:
            super().__init__(**options)
            created.append(options)

    monkeypatch.setattr(detectors, "Parallel", Recording)
    monkeypatch.setattr(
        inpaint_vessels, "get_slice_vessel_mask", lambda image: image > 0.5
    )
    volume = np.zeros((3, 3, 4))
    volume[0, 0, 3] = 1

    mask = detectors.axial_vessel_mask(volume)

    assert created == [{"n_jobs": -1}]
    assert mask.sum() == 1 and mask[0, 0, 3]


def test_frangi_runs_in_3d_with_the_pipelines_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    real = detectors.frangi

    def spy(image, **options):
        captured["shape"] = image.shape
        captured["options"] = options
        return real(image, **options)

    monkeypatch.setattr(detectors, "frangi", spy)
    volume = synthetic_volume(objects(12))

    detectors.frangi3d_vessel_mask(volume)

    assert captured["shape"] == volume.shape
    assert captured["options"] == {
        "sigmas": FRANGI_SIGMAS,
        "beta": FRANGI_BETA,
        "alpha": FRANGI_ALPHA,
        "black_ridges": FRANGI_BLACK_RIDGES,
    }


def test_vesselness_is_rescaled_by_a_percentile_not_the_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = {}

    class Spy:
        def __init__(self, **options) -> None:
            pass

        def fit(self, features: np.ndarray) -> "Spy":
            seen["features"] = features
            self.labels_ = np.zeros(len(features), dtype=int)
            return self

    # A degenerate voxel reaches 1 while real vessels respond around 1e-3.
    response = np.full((20, 20, 20), 1e-3)
    response[0, 0, 0] = 1.0
    monkeypatch.setattr(detectors, "frangi", lambda *args, **options: response)
    monkeypatch.setattr(detectors, "KMeans", Spy)
    volume = 0.3 + np.random.default_rng(0).random((20, 20, 20)) * 0.1

    detectors.frangi3d_vessel_mask(volume)

    vesselness = seen["features"][:, 0]
    assert vesselness.max() == 1.0
    assert np.median(vesselness) == pytest.approx(1.0)


def test_the_smaller_cluster_is_the_vessels(monkeypatch: pytest.MonkeyPatch) -> None:
    class TwoToOne:
        def __init__(self, **options) -> None:
            pass

        def fit(self, features: np.ndarray) -> "TwoToOne":
            self.labels_ = (np.arange(len(features)) < 10).astype(int)
            return self

    monkeypatch.setattr(detectors, "KMeans", TwoToOne)
    volume = 0.3 + np.random.default_rng(0).random((6, 6, 6)) * 0.1

    mask = detectors.frangi3d_vessel_mask(volume)

    assert mask.sum() == 10


@pytest.mark.parametrize(
    "volume",
    [np.zeros((6, 6, 6)), np.full((6, 6, 6), 0.5)],
    ids=["no-brain", "constant"],
)
def test_an_unusable_volume_has_no_vessels(volume: np.ndarray) -> None:
    mask = detectors.frangi3d_vessel_mask(volume)

    assert mask.shape == volume.shape and not mask.any()


def test_a_tube_has_more_linearity_than_a_blob_or_the_background() -> None:
    shapes = objects()
    linearity = detectors.tube_linearity(synthetic_volume(shapes))

    background = ~np.logical_or.reduce(list(shapes.values()))
    assert linearity[shapes["tube_z"]].mean() > 5 * linearity[shapes["blob"]].mean()
    assert linearity[shapes["tube_x"]].mean() > 5 * linearity[background].mean()


def test_tubes_in_any_direction_are_found_and_blobs_are_not(synthetic) -> None:
    shapes, mask = synthetic

    assert (mask & shapes["tube_z"]).sum() / shapes["tube_z"].sum() > 0.5
    assert (mask & shapes["tube_x"]).sum() / shapes["tube_x"].sum() > 0.5
    assert not (mask & shapes["blob"]).any()
