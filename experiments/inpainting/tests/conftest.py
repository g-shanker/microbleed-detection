import numpy as np
import pytest

from experiments.inpainting import detectors
from experiments.inpainting.tests.support import (
    AFFINE,
    LESION,
    SHAPE,
    THROUGH_PLANE,
    base_volume,
)
from microbleednet.core.datamodels import PreprocessInput, PreprocessOutput
from microbleednet.core.engines import processor
from microbleednet.core.transforms import inpaint_vessels


class SerialParallel:
    """Run joblib's delayed calls in order, without starting worker processes."""

    def __init__(self, **options) -> None:
        self.options = options

    def __call__(self, tasks):
        return [function(*args, **kwargs) for function, args, kwargs in tasks]


@pytest.fixture(autouse=True)
def serial_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests fast and in-process; the real pool spawns worker processes."""
    monkeypatch.setattr(detectors, "Parallel", SerialParallel)


def fake_frangi3d(volume: np.ndarray) -> np.ndarray:
    """Bright voxels, plus one voxel only a 3D detector would find."""
    mask = volume > 0.9
    mask[THROUGH_PLANE] = True
    return mask


@pytest.fixture
def fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> list[PreprocessInput]:
    """Replace FSL preprocessing and vessel detection with tiny deterministic ones.

    The fake preprocessing calls ``processor.inpaint_vessels.apply`` like the real
    one, so the experiment's capture of that call is exercised. Returns the
    inputs it was called with.
    """
    calls: list[PreprocessInput] = []

    def preprocess(preprocess_input: PreprocessInput) -> PreprocessOutput:
        calls.append(preprocess_input)
        volume = processor.inpaint_vessels.apply(base_volume())
        lesions = np.zeros(SHAPE, dtype=np.uint8)
        lesions[LESION] = 1
        return PreprocessOutput(
            volume=volume,
            mask=lesions,
            brain_mask=np.ones(SHAPE, dtype=np.uint8),
            affine=AFFINE,
            bounding_box=((0, SHAPE[0]), (0, SHAPE[1]), (0, SHAPE[2])),
            original_volume=preprocess_input.volume,
        )

    monkeypatch.setattr(processor, "preprocess", preprocess)
    monkeypatch.setattr(
        inpaint_vessels, "get_slice_vessel_mask", lambda image: image > 0.9
    )
    monkeypatch.setitem(detectors.DETECTORS, "frangi3d", fake_frangi3d)
    monkeypatch.setitem(detectors.DETECTORS, "frangi3d_range", fake_frangi3d)
    return calls
