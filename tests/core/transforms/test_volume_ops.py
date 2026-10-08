import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from microbleednet.core import utils
from microbleednet.core.transforms import volume_ops


@pytest.mark.parametrize("offset", [-1, 1])
def test_translate_shifts_without_wrapping(offset: int) -> None:
    volume = np.zeros((3, 3, 1))
    volume[1, 1, 0] = 1

    translated = volume_ops.translate(volume, offset, offset)

    assert translated[1 + offset, 1 + offset, 0] == 1
    assert translated.sum() == 1


def test_add_noise_requires_matching_shape() -> None:
    volume = np.ones((2, 2, 2))
    noise = np.full(volume.shape, 0.5)

    np.testing.assert_array_equal(
        volume_ops.add_noise(volume, noise),
        np.full(volume.shape, 1.5),
    )
    with pytest.raises(ValueError, match="shapes do not match"):
        volume_ops.add_noise(volume, np.ones((1, 1, 1)))


def test_blur_uses_supplied_sigma() -> None:
    volume = np.zeros((5, 5, 1))
    volume[2, 2, 0] = 1

    blurred = volume_ops.blur(volume, 1.0)

    assert 0 < blurred[2, 2, 0] < 1


def test_normalize_volume_scales_by_positive_maximum() -> None:
    normalized = volume_ops.normalize_volume(np.array([[[0.0, 2.0, 4.0]]]))

    np.testing.assert_array_equal(normalized, np.array([[[0.0, 0.5, 1.0]]]))


@pytest.mark.parametrize("maximum", [0.0, np.nan])
def test_normalize_volume_rejects_invalid_maximum(maximum: float) -> None:
    with pytest.raises(
        ValueError,
        match="cannot be normalized",
    ):
        volume_ops.normalize_volume(np.array([[[maximum]]]))


def test_invert_volume_preserves_zero_background() -> None:
    volume = np.array([[[0.0, 1.0, 3.0]]])

    inverted = volume_ops.invert_volume(volume)

    np.testing.assert_array_equal(inverted, np.array([[[0.0, 2.0, 0.0]]]))


def test_reorient_to_canonical_flips_negative_axis() -> None:
    volume = nib.Nifti1Image(
        np.arange(8, dtype=float).reshape((2, 2, 2)), np.diag([-1, 1, 1, 1])
    )

    canonical = volume_ops.reorient_to_canonical(volume)

    assert nib.aff2axcodes(canonical.affine) == ("R", "A", "S")


def test_adjust_affine_for_crop_translates_in_voxel_coordinates() -> None:
    affine = np.diag([2.0, 3.0, 4.0, 1.0])

    adjusted = volume_ops.adjust_affine_for_crop(affine, (1, 2, 3))

    np.testing.assert_array_equal(adjusted[:3, 3], np.array([2.0, 6.0, 12.0]))


def test_restore_cropped_volume_restores_shape_and_orientation() -> None:
    original_volume = nib.Nifti1Image(
        np.zeros((4, 3, 2), dtype=np.uint8),
        np.diag([-1.0, 1.0, 1.0, 1.0]),
    )
    cropped = np.ones((2, 2, 2), dtype=np.uint8)

    restored = volume_ops.restore_cropped_volume(
        cropped,
        ((1, 3), (1, 3), (0, 2)),
        original_volume,
    )

    assert restored.shape == (4, 3, 2)
    assert restored.get_data_dtype() == np.dtype(np.uint8)
    assert restored.get_fdata().sum() == cropped.sum()
    np.testing.assert_array_equal(restored.affine, original_volume.affine)


def test_get_bounding_box_returns_positive_extent() -> None:
    volume = np.zeros((3, 4, 5))
    volume[1:3, 2:4, 3:5] = 1

    assert volume_ops.get_bounding_box(volume) == (
        (1, 3),
        (2, 4),
        (3, 5),
    )


def test_get_bounding_box_rejects_empty_volume() -> None:
    with pytest.raises(ValueError, match="bounding box"):
        volume_ops.get_bounding_box(np.zeros((2, 2, 2)))


def test_add_to_bounding_box_updates_only_bounded_region() -> None:
    volume = np.zeros((3, 3, 3))
    bounding_box = ((1, 3), (0, 2), (1, 3))

    volume_ops.add_to_bounding_box(volume, np.ones((2, 2, 2)), bounding_box)
    volume_ops.add_to_bounding_box(volume, 1, bounding_box)

    expected = np.zeros_like(volume)
    expected[1:3, 0:2, 1:3] = 2
    np.testing.assert_array_equal(volume, expected)


def test_extract_brain_requires_fsldir(monkeypatch) -> None:
    monkeypatch.delenv("FSLDIR", raising=False)
    volume = nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4))

    with pytest.raises(ValueError, match="FSLDIR is not set"):
        volume_ops.extract_brain(volume)


def test_extract_brain_requires_valid_fsldir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FSLDIR", str(tmp_path / "missing"))
    volume = nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4))

    with pytest.raises(ValueError, match="requires FSL"):
        volume_ops.extract_brain(volume)


def test_extract_brain_requires_bet_executable(tmp_path: Path, monkeypatch) -> None:
    fsldir = tmp_path / "fsl"
    fsldir.mkdir()
    monkeypatch.setenv("FSLDIR", str(fsldir))
    volume = nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4))

    with pytest.raises(ValueError, match="BET executable not found"):
        volume_ops.extract_brain(volume)


def test_extract_brain_runs_bet_and_materializes_output(
    tmp_path: Path, monkeypatch
) -> None:
    fsldir = tmp_path / "fsl"
    (fsldir / "bin").mkdir(parents=True)
    (fsldir / "bin" / "bet").touch()
    monkeypatch.setenv("FSLDIR", str(fsldir))
    calls = []

    def fake_run(command: list[str], check: bool) -> None:
        calls.append((command, check))
        shutil.copyfile(command[1], command[2])
        input_volume = volume_ops.io.load_volume(command[1])
        mask_path = Path(command[2]).with_name("post_bet_mask.nii.gz")
        nib.save(
            nib.Nifti1Image(
                (input_volume.get_fdata() > 0).astype(np.uint8),
                input_volume.affine,
            ),
            mask_path,
        )

    monkeypatch.setattr(utils.subprocess, "run", fake_run)
    volume = nib.Nifti1Image(
        np.arange(8, dtype=float).reshape((2, 2, 2)), np.eye(4)
    )

    extracted, brain_mask = volume_ops.extract_brain(volume)

    assert calls[0][0][0] == str(fsldir / "bin" / "bet")
    assert calls[0][0][-1] == "-m"
    assert calls[0][1] is True
    np.testing.assert_array_equal(extracted.get_fdata(), volume.get_fdata())
    np.testing.assert_array_equal(brain_mask.get_fdata(), volume.get_fdata() > 0)


@pytest.mark.parametrize(
    ("process_error", "message"),
    [
        (FileNotFoundError("missing"), "Could not start FSL BET"),
        (
            subprocess.CalledProcessError(2, "bet"),
            "FSL BET failed",
        ),
    ],
)
def test_extract_brain_translates_bet_failures(
    tmp_path: Path,
    monkeypatch,
    process_error: Exception,
    message: str,
) -> None:
    fsldir = tmp_path / "fsl"
    bet_path = fsldir / "bin" / "bet"
    bet_path.parent.mkdir(parents=True)
    bet_path.touch()
    monkeypatch.setenv("FSLDIR", str(fsldir))
    monkeypatch.setattr(
        utils.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(process_error),
    )
    volume = nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4))

    with pytest.raises(ValueError, match=message):
        volume_ops.extract_brain(volume)


def test_bias_field_correct_fast_runs_fast_and_preserves_geometry(
    tmp_path: Path, monkeypatch
) -> None:
    fsldir = tmp_path / "fsl"
    (fsldir / "bin").mkdir(parents=True)
    (fsldir / "bin" / "fast").touch()
    monkeypatch.setenv("FSLDIR", str(fsldir))
    calls = []

    def fake_run(command: list[str], check: bool) -> None:
        calls.append((command, check))
        input_volume = volume_ops.io.load_volume(command[-1])
        restored_path = Path(f"{command[3]}_restore.nii.gz")
        nib.save(
            nib.Nifti1Image(input_volume.get_fdata() + 1, input_volume.affine),
            restored_path,
        )

    monkeypatch.setattr(utils.subprocess, "run", fake_run)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    volume = nib.Nifti1Image(np.ones((2, 2, 2)), affine)

    corrected = volume_ops.bias_field_correct_fast(volume)

    command, check = calls[0]
    assert command[0] == str(fsldir / "bin" / "fast")
    assert command[1:3] == ["-B", "-o"]
    assert Path(command[3]).name == "fast"
    assert Path(command[4]).name == "pre_fast.nii.gz"
    assert check is True
    np.testing.assert_array_equal(corrected.get_fdata(), np.full((2, 2, 2), 2.0))
    np.testing.assert_array_equal(corrected.affine, affine)


def test_bias_field_correct_n4_uses_bet_mask_and_preserves_geometry(
    monkeypatch,
) -> None:
    sitk_images = []
    executions = []

    class FakeSimpleITKImage:
        def __init__(self, array):
            self.array = array
            self.spacing = None

        def SetSpacing(self, spacing):
            self.spacing = tuple(spacing)

    def get_image_from_array(array):
        image = FakeSimpleITKImage(array)
        sitk_images.append(image)
        return image

    class FakeN4Filter:
        def Execute(self, image, mask):
            executions.append((image, mask))
            return image

    monkeypatch.setitem(
        sys.modules,
        "SimpleITK",
        SimpleNamespace(
            GetImageFromArray=get_image_from_array,
            N4BiasFieldCorrectionImageFilter=FakeN4Filter,
            GetArrayFromImage=lambda image: image.array + 2,
        ),
    )
    volume_data = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    affine = np.diag([1.25, 2.5, 3.75, 1.0])
    volume = nib.Nifti1Image(volume_data, affine)
    bet_mask_data = np.zeros((2, 3, 4), dtype=np.uint8)
    bet_mask_data[0, 0, 0] = 1
    bet_mask_data[1, 2, 3] = 1
    bet_mask = nib.Nifti1Image(bet_mask_data, affine)

    corrected = volume_ops.bias_field_correct_n4(volume, bet_mask)

    np.testing.assert_array_equal(sitk_images[0].array, volume_data.T)
    np.testing.assert_array_equal(sitk_images[1].array, bet_mask_data.T)
    expected_spacing = tuple(float(value) for value in volume.header.get_zooms()[:3])
    assert sitk_images[0].spacing == expected_spacing
    assert sitk_images[1].spacing == expected_spacing
    assert executions[0][0] is sitk_images[0]
    assert executions[0][1] is sitk_images[1]
    np.testing.assert_array_equal(corrected.get_fdata(), volume_data + 2)
    np.testing.assert_array_equal(corrected.affine, affine)