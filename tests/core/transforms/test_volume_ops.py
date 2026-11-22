import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import nibabel as nib
import numpy as np
import pytest

from microbleednet.core import io, utils
from microbleednet.core.transforms import volume_ops
from tests.support_core import image, voxels

BOX = ((1, 3), (1, 3), (0, 2))


@pytest.fixture
def install_fsl(tmp_path: Path, monkeypatch):
    """Point FSLDIR at a fake install containing the requested executable."""

    def install(tool: str) -> Path:
        executable = tmp_path / "fsl" / "bin" / tool
        executable.parent.mkdir(parents=True)
        executable.touch()
        monkeypatch.setenv("FSLDIR", str(tmp_path / "fsl"))
        return executable

    return install


@pytest.mark.parametrize("offset", [-1, 1])
def test_translate_shifts_without_wrapping(offset: int) -> None:
    translated = volume_ops.translate(voxels((3, 3, 1), (1, 1, 0)), offset, offset)

    np.testing.assert_array_equal(
        translated, voxels((3, 3, 1), (1 + offset, 1 + offset, 0))
    )


def test_add_noise_adds_matching_noise() -> None:
    result = volume_ops.add_noise(np.ones((2, 2, 2)), np.full((2, 2, 2), 0.5))

    np.testing.assert_array_equal(result, np.full((2, 2, 2), 1.5))


def test_add_noise_rejects_mismatched_shape() -> None:
    with pytest.raises(ValueError, match="shapes do not match"):
        volume_ops.add_noise(np.ones((2, 2, 2)), np.ones((1, 1, 1)))


def test_blur_spreads_intensity_by_sigma() -> None:
    blurred = volume_ops.blur(voxels((5, 5, 1), (2, 2, 0)), 1.0)

    assert 0 < blurred[2, 2, 0] < 1
    assert blurred[1, 2, 0] > 0


def test_normalize_volume_scales_by_positive_maximum() -> None:
    normalized = volume_ops.normalize_volume(np.array([[[0.0, 2.0, 4.0]]]))

    np.testing.assert_array_equal(normalized, np.array([[[0.0, 0.5, 1.0]]]))


@pytest.mark.parametrize("maximum", [0.0, np.nan])
def test_normalize_volume_rejects_invalid_maximum(maximum: float) -> None:
    with pytest.raises(ValueError, match="cannot be normalized"):
        volume_ops.normalize_volume(np.array([[[maximum]]]))


def test_invert_volume_preserves_zero_background() -> None:
    inverted = volume_ops.invert_volume(np.array([[[0.0, 1.0, 3.0]]]))

    np.testing.assert_array_equal(inverted, np.array([[[0.0, 2.0, 0.0]]]))


def test_reorient_to_canonical_flips_negative_axis() -> None:
    flipped = image(
        data=np.arange(8, dtype=float).reshape((2, 2, 2)),
        affine=np.diag([-1, 1, 1, 1]),
    )

    canonical = volume_ops.reorient_to_canonical(flipped)

    assert nib.aff2axcodes(canonical.affine) == ("R", "A", "S")


def test_adjust_affine_for_crop_translates_in_voxel_coordinates() -> None:
    affine = np.diag([2.0, 3.0, 4.0, 1.0])

    adjusted = volume_ops.adjust_affine_for_crop(affine, (1, 2, 3))

    np.testing.assert_array_equal(adjusted[:3, 3], [2.0, 6.0, 12.0])


@pytest.mark.parametrize(
    ("affine", "dtype", "value"),
    [
        (np.diag([-1.0, 1.0, 1.0, 1.0]), np.uint8, 1),
        (np.eye(4), np.float32, 0.25),
    ],
    ids=["flipped-uint8", "float-probabilities"],
)
def test_restore_cropped_volume_restores_shape_dtype_and_geometry(
    affine: np.ndarray, dtype: type, value: float
) -> None:
    original = image((4, 3, 2), 0, affine=affine, dtype=np.uint8)
    cropped = np.full((2, 2, 2), value, dtype=dtype)

    restored = volume_ops.restore_cropped_volume(cropped, BOX, original)

    assert restored.shape == (4, 3, 2)
    assert restored.get_data_dtype() == np.dtype(dtype)
    assert np.count_nonzero(restored.get_fdata()) == cropped.size
    assert np.count_nonzero(restored.get_fdata() == value) == cropped.size
    np.testing.assert_array_equal(restored.affine, affine)


def test_restore_cropped_volume_round_trips_full_canonical_volume() -> None:
    data = np.arange(1, 25, dtype=np.float32).reshape((4, 3, 2))
    original = image(data=data, affine=np.diag([-1.0, 1.0, -1.0, 1.0]))
    canonical = volume_ops.reorient_to_canonical(original)
    full_box = (
        (0, canonical.shape[0]),
        (0, canonical.shape[1]),
        (0, canonical.shape[2]),
    )

    restored = volume_ops.restore_cropped_volume(
        np.asarray(canonical.dataobj), full_box, original
    )

    np.testing.assert_array_equal(restored.get_fdata(), data)


def test_get_bounding_box_returns_positive_extent() -> None:
    volume = voxels((3, 4, 5), (slice(1, 3), slice(2, 4), slice(3, 5)))

    assert volume_ops.get_bounding_box(volume) == ((1, 3), (2, 4), (3, 5))


def test_get_bounding_box_rejects_empty_volume() -> None:
    with pytest.raises(ValueError, match="bounding box"):
        volume_ops.get_bounding_box(np.zeros((2, 2, 2)))


def test_add_to_bounding_box_updates_only_bounded_region() -> None:
    volume = np.zeros((3, 3, 3))
    box = ((1, 3), (0, 2), (1, 3))

    volume_ops.add_to_bounding_box(volume, np.ones((2, 2, 2)), box)
    volume_ops.add_to_bounding_box(volume, 1, box)

    np.testing.assert_array_equal(
        volume, voxels((3, 3, 3), (slice(1, 3), slice(0, 2), slice(1, 3)), value=2)
    )


@pytest.mark.parametrize(
    ("fsldir", "message"),
    [
        (None, "FSLDIR is not set"),
        ("missing", "requires FSL"),
        ("empty", "BET executable not found"),
    ],
)
def test_extract_brain_rejects_unusable_fsl(
    tmp_path: Path, monkeypatch, fsldir: str | None, message: str
) -> None:
    if fsldir is None:
        monkeypatch.delenv("FSLDIR", raising=False)
    else:
        if fsldir == "empty":
            (tmp_path / fsldir).mkdir()
        monkeypatch.setenv("FSLDIR", str(tmp_path / fsldir))

    with pytest.raises(ValueError, match=message):
        volume_ops.extract_brain(image())


def test_extract_brain_runs_bet_and_materializes_output(
    install_fsl, monkeypatch
) -> None:
    bet = install_fsl("bet")
    calls = []

    def fake_run(command: list[str], check: bool) -> None:
        calls.append((command, check))
        source = io.load_volume(command[1])
        nib.save(source, command[2])
        nib.save(
            image(data=(source.get_fdata() > 0).astype(np.uint8), affine=source.affine),
            Path(command[2]).with_name("post_bet_mask.nii.gz"),
        )

    monkeypatch.setattr(utils.subprocess, "run", fake_run)
    volume = image(data=np.arange(8, dtype=float).reshape((2, 2, 2)))

    extracted, brain_mask = volume_ops.extract_brain(volume)

    command, check = calls[0]
    assert command[0] == str(bet)
    assert command[-1] == "-m"
    assert check is True
    np.testing.assert_array_equal(extracted.get_fdata(), volume.get_fdata())
    np.testing.assert_array_equal(brain_mask.get_fdata(), volume.get_fdata() > 0)


@pytest.mark.parametrize(
    ("process_error", "message"),
    [
        (FileNotFoundError("missing"), "Could not start FSL BET"),
        (subprocess.CalledProcessError(2, "bet"), "FSL BET failed"),
    ],
)
def test_extract_brain_translates_bet_failures(
    install_fsl, monkeypatch, process_error: Exception, message: str
) -> None:
    install_fsl("bet")
    monkeypatch.setattr(utils.subprocess, "run", Mock(side_effect=process_error))

    with pytest.raises(ValueError, match=message):
        volume_ops.extract_brain(image())


def test_bias_field_correct_fast_runs_fast_and_preserves_geometry(
    install_fsl, monkeypatch
) -> None:
    fast = install_fsl("fast")
    calls = []

    def fake_run(command: list[str], check: bool) -> None:
        calls.append((command, check))
        input_volume = io.load_volume(command[-1])
        nib.save(
            image(data=input_volume.get_fdata() + 1, affine=input_volume.affine),
            f"{command[3]}_restore.nii.gz",
        )

    monkeypatch.setattr(utils.subprocess, "run", fake_run)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])

    corrected = volume_ops.bias_field_correct_fast(image(affine=affine))

    command, check = calls[0]
    assert command[0] == str(fast)
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

    class FakeImage:
        def __init__(self, array: np.ndarray) -> None:
            self.array = array
            self.spacing: tuple[float, ...] | None = None
            sitk_images.append(self)

        def SetSpacing(self, spacing) -> None:
            self.spacing = tuple(spacing)

    class FakeN4Filter:
        def Execute(self, volume: FakeImage, mask: FakeImage) -> FakeImage:
            executions.append((volume, mask))
            return volume

    monkeypatch.setitem(
        sys.modules,
        "SimpleITK",
        SimpleNamespace(
            GetImageFromArray=FakeImage,
            N4BiasFieldCorrectionImageFilter=FakeN4Filter,
            GetArrayFromImage=lambda fake_image: fake_image.array + 2,
        ),
    )
    volume_data = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    affine = np.diag([1.25, 2.5, 3.75, 1.0])
    bet_mask_data = voxels((2, 3, 4), (0, 0, 0), (1, 2, 3), dtype=np.uint8)
    volume = image(data=volume_data, affine=affine)

    corrected = volume_ops.bias_field_correct_n4(
        volume, image(data=bet_mask_data, affine=affine)
    )

    np.testing.assert_array_equal(sitk_images[0].array, volume_data.T)
    np.testing.assert_array_equal(sitk_images[1].array, bet_mask_data.T)
    expected_spacing = tuple(float(value) for value in volume.header.get_zooms()[:3])
    assert [item.spacing for item in sitk_images] == [expected_spacing] * 2
    assert executions == [(sitk_images[0], sitk_images[1])]
    np.testing.assert_array_equal(corrected.get_fdata(), volume_data + 2)
    np.testing.assert_array_equal(corrected.affine, affine)
