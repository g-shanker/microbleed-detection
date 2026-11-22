from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import nibabel as nib
import numpy as np
import pytest
import torch
import torch.nn as nn
from skimage.measure._regionprops import RegionProperties

from microbleednet.core import utils
from microbleednet.core.datamodels import (
    BiasFieldCorrection,
    Modality,
    PreprocessInput,
)
from microbleednet.core.engines import inference, processor
from tests.support_core import image, voxels


def identity(value):
    return value


def preprocess(
    volume: nib.Nifti1Image,
    mask: nib.Nifti1Image | None = None,
    modality: str = "QSM",
    bias_field_correction: BiasFieldCorrection = "fast",
):
    return processor.preprocess(
        PreprocessInput(volume, mask, cast(Modality, modality), bias_field_correction)
    )


@pytest.fixture
def steps(monkeypatch) -> SimpleNamespace:
    """Stub the FSL, N4, and vessel-inpainting steps; keep array transforms real."""
    stubs = SimpleNamespace(
        extract_brain=Mock(
            side_effect=lambda volume: (volume, image(volume.shape, dtype=np.uint8))
        ),
        fast=Mock(side_effect=identity),
        n4=Mock(side_effect=lambda volume, brain_mask: volume),
        invert=Mock(wraps=processor.volume_ops.invert_volume),
    )
    ops = processor.volume_ops
    monkeypatch.setattr(ops, "extract_brain", stubs.extract_brain)
    monkeypatch.setattr(ops, "bias_field_correct_fast", stubs.fast)
    monkeypatch.setattr(ops, "bias_field_correct_n4", stubs.n4)
    monkeypatch.setattr(ops, "invert_volume", stubs.invert)
    monkeypatch.setattr(processor.inpaint_vessels, "apply", identity)
    return stubs


def test_preprocess_qsm_skips_contrast_operations(steps) -> None:
    preprocess(image())

    steps.fast.assert_not_called()
    steps.n4.assert_not_called()
    steps.invert.assert_not_called()


def test_preprocess_returns_normalized_volume_mask_and_geometry(steps) -> None:
    output = preprocess(image(value=4), image(dtype=np.uint8))

    assert output.mask is not None
    assert output.mask.dtype == np.uint8
    np.testing.assert_array_equal(output.mask, np.ones((2, 2, 2)))
    np.testing.assert_array_equal(output.brain_mask, np.ones((2, 2, 2)))
    np.testing.assert_array_equal(output.volume, np.ones((2, 2, 2)))
    np.testing.assert_array_equal(output.affine, np.eye(4))
    assert output.bounding_box == ((0, 2), (0, 2), (0, 2))


def test_preprocess_without_mask_returns_no_mask(steps) -> None:
    assert preprocess(image()).mask is None


def test_preprocess_crops_to_brain_mask_and_shifts_affine(steps) -> None:
    inside = (slice(1, 4),) * 3
    volume_data = voxels((4, 4, 4), inside, value=-1)
    volume_data[2, 2, 2] = 2
    brain_mask = voxels((4, 4, 4), inside, dtype=np.uint8)
    steps.extract_brain.side_effect = lambda _: (
        image(data=volume_data),
        image(data=brain_mask),
    )

    output = preprocess(image(data=volume_data))

    assert output.bounding_box == ((1, 4), (1, 4), (1, 4))
    assert output.volume.shape == (3, 3, 3)
    np.testing.assert_array_equal(output.brain_mask, np.ones((3, 3, 3)))
    np.testing.assert_array_equal(output.affine[:3, 3], [1, 1, 1])


@pytest.mark.parametrize(
    ("bias_field_correction", "used", "unused"),
    [("fast", "fast", "n4"), ("n4", "n4", "fast")],
)
def test_preprocess_swi_applies_selected_bias_correction_and_inverts(
    steps, bias_field_correction: BiasFieldCorrection, used: str, unused: str
) -> None:
    volume, bet_mask = image(), image(dtype=np.uint8)
    steps.extract_brain.side_effect = lambda _: (volume, bet_mask)

    preprocess(volume, None, "SWI", bias_field_correction)

    getattr(steps, unused).assert_not_called()
    getattr(steps, used).assert_called_once_with(
        *((volume,) if used == "fast" else (volume, bet_mask))
    )
    steps.invert.assert_called_once()


@pytest.mark.parametrize(
    ("volume", "mask", "message"),
    [
        (image(), image(affine=np.diag([2.0, 1, 1, 1])), "affines do not match"),
        (image((2, 2, 2, 1)), image((2, 2, 2, 1)), "3D volume"),
        (image(), image((1, 2, 2)), "shapes do not match"),
        (image(value=np.nan), image(), "non-finite values"),
        (image(), image(value=np.nan), "non-finite values"),
        (image(value=0), image(), "no positive voxels"),
        (
            image(),
            image(data=voxels((2, 2, 2), (0, 0, 0), value=2, dtype=np.uint8)),
            "not binary",
        ),
    ],
    ids=[
        "affine",
        "not-3d",
        "mask-shape",
        "nan-volume",
        "nan-mask",
        "empty-volume",
        "non-binary-mask",
    ],
)
def test_preprocess_rejects_invalid_inputs(volume, mask, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        preprocess(volume, mask)


def test_preprocess_rejects_invalid_voxel_spacing() -> None:
    volume = image()
    volume.header.set_zooms((0.0, 1.0, 1.0))

    with pytest.raises(ValueError, match="voxel spacing is invalid"):
        preprocess(volume, image())


def test_predict_logits_runs_batched_model_in_eval_mode() -> None:
    model = nn.Conv3d(2, 2, kernel_size=1)
    volume = np.stack((np.ones((2, 2, 2)), np.zeros((2, 2, 2))))

    result = utils.predict_logits(model, volume, use_amp=False)

    assert result.shape == (2, 2, 2, 2)
    assert not model.training


def test_predict_logits_enables_cuda_autocast_when_requested(monkeypatch) -> None:
    autocast_calls = []
    monkeypatch.setattr(utils, "get_model_device", lambda _: torch.device("cuda"))
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(
        utils.torch,
        "autocast",
        lambda **kwargs: autocast_calls.append(kwargs) or nullcontext(),
    )

    result = utils.predict_logits(
        nn.Identity(), np.ones((2, 2, 2, 2), dtype=np.float32), use_amp=True
    )

    assert result.shape == (2, 2, 2, 2)
    assert result.dtype == torch.float32
    assert autocast_calls == [
        {"device_type": "cuda", "dtype": torch.bfloat16, "enabled": True}
    ]


def test_detector_probability_uses_volume_and_frst_channels(monkeypatch) -> None:
    logits = torch.zeros((2, 2, 2, 2))
    logits[1, 0, 0, 0] = 10
    monkeypatch.setattr(utils, "predict_logits", lambda model, volume, use_amp: logits)

    probability_map = inference.infer_detector(
        nn.Conv3d(2, 2, kernel_size=1),
        np.ones((2, 2, 2)),
        np.zeros((2, 2, 2)),
        use_amp=False,
    )

    assert probability_map.shape == (2, 2, 2)
    assert probability_map[0, 0, 0] > 0.5


def test_detector_probability_tiles_large_volumes(monkeypatch) -> None:
    tile_shapes = []
    monkeypatch.setattr(inference, "DETECTOR_INFERENCE_TILE_SIZE", 4)
    monkeypatch.setattr(inference, "DETECTOR_INFERENCE_TILE_OVERLAP", 2)

    def predict_logits(_model, model_input, _use_amp):
        tile_shapes.append(model_input.shape)
        logits = torch.zeros((2, *model_input.shape[1:]))
        logits[1] = 10
        return logits

    monkeypatch.setattr(utils, "predict_logits", predict_logits)

    probability_map = inference.infer_detector(
        nn.Identity(), np.ones((3, 6, 5)), np.zeros((3, 6, 5)), use_amp=True
    )

    assert probability_map.shape == (3, 6, 5)
    assert np.all(probability_map > 0.5)
    assert len(tile_shapes) == 4
    assert all(shape[0] == 2 for shape in tile_shapes)
    assert all(max(shape[1:]) <= 4 for shape in tile_shapes)


class Student(nn.Module):
    """Accept the first candidate patch and reject the second."""

    def __init__(self) -> None:
        super().__init__()
        self.device_parameter = nn.Parameter(torch.zeros(1))
        self.batch_sizes: list[int] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(inputs.shape[0])
        accepted = len(self.batch_sizes) == 1
        return torch.tensor([[0.0, 2.0] if accepted else [2.0, 0.0]])


def infer_discriminator(model: nn.Module, detector_probability: np.ndarray):
    ones = np.ones_like(detector_probability)
    return inference.infer_discriminator(
        model,
        ones,
        ones,
        detector_probability,
        detector_threshold=0.5,
        patch_size=2,
        discriminator_threshold=0.5,
        use_amp=False,
    )


def test_infer_discriminator_returns_retained_candidate_volume() -> None:
    student = Student()
    candidates = voxels((3, 3, 3), (0, 0, 0), (2, 2, 2), value=0.9)

    output = infer_discriminator(student, candidates)

    np.testing.assert_array_equal(output, voxels((3, 3, 3), (0, 0, 0), dtype=np.uint8))
    assert student.batch_sizes == [1, 1]


def test_infer_discriminator_returns_empty_volume_without_candidates() -> None:
    output = infer_discriminator(nn.Conv3d(2, 2, kernel_size=1), np.zeros((3, 3, 3)))

    np.testing.assert_array_equal(output, np.zeros((3, 3, 3), dtype=np.uint8))


def candidate_components() -> dict[str, np.ndarray]:
    """Isolated components inside a brain whose one-voxel border is background."""
    shape = (15, 15, 15)
    return {
        "core": voxels(shape, (slice(6, 9),) * 3, dtype=bool),
        "edge": voxels(shape, (slice(1, 4),) * 3, dtype=bool),
        "line": voxels(shape, (11, 4, slice(3, 8)), dtype=bool),
        "speck": voxels(shape, (11, 11, 11), dtype=bool),
    }


@pytest.mark.parametrize(
    ("filters", "kept"),
    [
        ({}, {"core", "edge", "line", "speck"}),
        ({"minimum_volume_mm3": 2.0}, {"core", "edge", "line"}),
        ({"maximum_ellipticity": 0.5}, {"core", "edge", "speck"}),
        ({"minimum_brain_distance_mm": 3.0}, {"core", "line", "speck"}),
    ],
    ids=["none", "volume", "ellipticity", "boundary"],
)
def test_postprocess_removes_components_failing_each_filter(
    filters: dict[str, float], kept: set[str]
) -> None:
    components = candidate_components()
    brain_mask = voxels((15, 15, 15), (slice(1, 14),) * 3, dtype=np.uint8)
    candidates = np.logical_or.reduce(list(components.values())).astype(np.uint8)
    limits = {
        "minimum_volume_mm3": 0.0,
        "maximum_ellipticity": 1.0,
        "minimum_brain_distance_mm": 0.0,
    } | filters

    output = processor.postprocess(candidates, brain_mask, (1.0, 1.0, 1.0), **limits)

    expected = np.logical_or.reduce([components[name] for name in kept])
    np.testing.assert_array_equal(output, expected.astype(np.uint8))


@pytest.mark.parametrize(
    ("eigenvalues", "expected"),
    [([], 0.0), ([0.0, 0.0, 0.0], 0.0), ([1.0, 2.0, 4.0], 0.75)],
)
def test_component_ellipticity_from_eigenvalues(
    eigenvalues: list[float], expected: float
) -> None:
    region = SimpleNamespace(inertia_tensor_eigvals=np.array(eigenvalues))

    assert processor.component_ellipticity(
        cast(RegionProperties, region)
    ) == pytest.approx(expected)
