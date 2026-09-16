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
from microbleednet.core.engines import inference, processor


def _volume(shape: tuple[int, int, int] = (2, 2, 2), affine=None) -> nib.Nifti1Image:
    return nib.Nifti1Image(np.ones(shape), np.eye(4) if affine is None else affine)


def _stub_processing_steps(monkeypatch) -> tuple[Mock, Mock]:
    monkeypatch.setattr(
        processor.volume_ops, "reorient_to_canonical", lambda value: value
    )
    monkeypatch.setattr(processor.volume_ops, "extract_brain", lambda value: value)
    monkeypatch.setattr(processor.volume_ops, "normalize_volume", lambda value: value)
    monkeypatch.setattr(
        processor.volume_ops,
        "tight_crop_volume",
        lambda value: (value, ((0, 2), (0, 2), (0, 2))),
    )
    monkeypatch.setattr(processor.inpaint_vessels, "apply", lambda value: value)
    bias_correct = Mock(side_effect=lambda value: value)
    invert = Mock(side_effect=lambda value: value)
    monkeypatch.setattr(processor.volume_ops, "bias_field_correct_n4", bias_correct)
    monkeypatch.setattr(processor.volume_ops, "invert_volume", invert)
    return bias_correct, invert


def test_preprocess_qsm_skips_contrast_operations(monkeypatch) -> None:
    bias_correct, invert = _stub_processing_steps(monkeypatch)
    mask = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))

    result = processor.preprocess(_volume(), mask, "QSM")

    assert result.mask.dtype == np.uint8
    np.testing.assert_array_equal(result.image, np.ones((2, 2, 2)))
    np.testing.assert_array_equal(result.affine, np.eye(4))
    bias_correct.assert_not_called()
    invert.assert_not_called()


def test_preprocess_swi_processes_and_crops_mask(monkeypatch) -> None:
    bias_correct, invert = _stub_processing_steps(monkeypatch)
    mask = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))

    result = processor.preprocess(_volume(), mask, "SWI")

    assert result.mask is not None
    np.testing.assert_array_equal(result.mask, np.ones((2, 2, 2), dtype=int))
    assert result.mask.dtype == np.uint8
    bias_correct.assert_called_once()
    invert.assert_called_once()


def test_preprocess_rejects_mask_with_different_affine() -> None:
    mask = _volume(affine=np.diag([2.0, 1.0, 1.0, 1.0]))

    with pytest.raises(ValueError, match="affines do not match"):
        processor.preprocess(_volume(), mask, "QSM")


def test_preprocess_rejects_reoriented_mask_with_different_shape() -> None:
    with pytest.raises(ValueError, match="shapes do not match"):
        processor.preprocess(_volume(), _volume((1, 2, 2)), "QSM")


def test_preprocess_rejects_invalid_voxel_spacing() -> None:
    volume = _volume()
    volume.header.set_zooms((0.0, 1.0, 1.0))

    with pytest.raises(ValueError, match="voxel spacing must be positive"):
        processor.preprocess(volume, _volume(), "QSM")


def test_preprocess_rejects_non_finite_volume_data() -> None:
    volume = nib.Nifti1Image(np.full((2, 2, 2), np.nan), np.eye(4))

    with pytest.raises(ValueError, match="image contains non-finite values"):
        processor.preprocess(volume, _volume(), "QSM")


def test_preprocess_rejects_empty_volume() -> None:
    volume = nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))

    with pytest.raises(ValueError, match="image is empty"):
        processor.preprocess(volume, _volume(), "QSM")


def test_predict_logits_builds_batched_volume_on_model_device() -> None:
    model = nn.Conv3d(2, 2, kernel_size=1)

    volume = np.stack((np.ones((2, 2, 2)), np.zeros((2, 2, 2))))
    result = utils.predict_logits(model, volume)

    assert result.shape == (2, 2, 2, 2)
    assert not model.training


def test_predict_logits_uses_persisted_frst_channel() -> None:
    model = nn.Conv3d(2, 2, kernel_size=1)

    volume = np.stack((np.ones((2, 2, 2)), np.zeros((2, 2, 2))))
    result = utils.predict_logits(model, volume)

    assert result.shape == (2, 2, 2, 2)


def test_detector_probability_uses_volume_and_frst_channels(monkeypatch) -> None:
    logits = torch.zeros((2, 2, 2, 2))
    logits[1, 0, 0, 0] = 10
    monkeypatch.setattr(utils, "predict_logits", lambda model, volume: logits)

    probability_map = inference.infer_detector(
        nn.Conv3d(2, 2, kernel_size=1),
        np.ones((2, 2, 2)),
        np.zeros((2, 2, 2)),
    )

    assert probability_map.shape == (2, 2, 2)
    assert probability_map[0, 0, 0] > 0.5


def test_infer_discriminator_returns_retained_candidate_volume() -> None:
    detector_probability = np.zeros((3, 3, 3))
    detector_probability[0, 0, 0] = 0.9
    detector_probability[2, 2, 2] = 0.9
    volume = np.ones((3, 3, 3))
    frst = np.ones((3, 3, 3))

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.device_parameter = nn.Parameter(torch.zeros(1))
            self.batch_sizes = []

        def forward(self, inputs):
            self.batch_sizes.append(inputs.shape[0])
            if len(self.batch_sizes) == 1:
                return torch.tensor([[0.0, 2.0]])
            return torch.tensor([[2.0, 0.0]])

    student = Student()
    output = inference.infer_discriminator(
        student,
        volume,
        frst,
        detector_probability,
        detector_threshold=0.5,
        patch_size=2,
        discriminator_threshold=0.5,
    )

    assert output[0, 0, 0] == 1
    assert output[2, 2, 2] == 0
    assert student.batch_sizes == [1, 1]


def test_infer_discriminator_returns_empty_volume_without_candidates() -> None:
    probability = np.zeros((3, 3, 3))

    output = inference.infer_discriminator(
        nn.Conv3d(2, 2, kernel_size=1),
        np.ones_like(probability),
        np.ones_like(probability),
        probability,
        detector_threshold=0.5,
        patch_size=2,
        discriminator_threshold=0.5,
    )

    np.testing.assert_array_equal(output, np.zeros_like(probability, dtype=np.uint8))


def test_postprocess_rejects_elliptical_components(monkeypatch) -> None:
    candidate_mask = np.ones((3, 3, 3), dtype=np.uint8)
    monkeypatch.setattr(processor, "component_ellipticity", lambda _: 1.0)

    output = processor.postprocess(
        candidate_mask,
        np.ones_like(candidate_mask, dtype=float),
        (1.0, 1.0, 1.0),
        minimum_volume_mm3=0.0,
        maximum_ellipticity=0.2,
        minimum_brain_distance_mm=0.0,
    )

    assert not output.any()


def test_postprocess_rejects_components_near_brain_boundary(monkeypatch) -> None:
    candidate_mask = np.ones((3, 3, 3), dtype=np.uint8)
    monkeypatch.setattr(
        processor,
        "distance_transform_edt",
        lambda *args, **kwargs: np.zeros_like(candidate_mask),
    )

    output = processor.postprocess(
        candidate_mask,
        np.ones_like(candidate_mask, dtype=float),
        (1.0, 1.0, 1.0),
        minimum_volume_mm3=0.0,
        maximum_ellipticity=1.0,
        minimum_brain_distance_mm=1.0,
    )

    assert not output.any()


def test_component_ellipticity_handles_empty_eigenvalues() -> None:
    region = SimpleNamespace(inertia_tensor_eigvals=np.array([]))

    assert processor.component_ellipticity(cast(RegionProperties, region)) == 0.0