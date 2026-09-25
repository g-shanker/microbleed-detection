import nibabel as nib
import numpy as np
import pytest

from microbleednet.constants import DETECTOR_PATCH_SIZE, DISCRIMINATOR_PATCH_SIZE
from microbleednet.core.datamodels import PreprocessInput


def test_patch_sizes_are_fixed_model_input_contracts() -> None:
    assert DETECTOR_PATCH_SIZE == 48
    assert DISCRIMINATOR_PATCH_SIZE == 24


@pytest.mark.parametrize(
    ("volume_affine", "mask_affine", "message"),
    [
        (None, np.eye(4), "Volume affine is missing"),
        (np.full((4, 4), np.nan), np.eye(4), "Volume affine is not finite"),
        (np.eye(4), None, "Mask affine is missing"),
        (np.eye(4), np.full((4, 4), np.nan), "Mask affine is not finite"),
    ],
)
def test_preprocess_input_rejects_invalid_affines(
    volume_affine, mask_affine, message: str
) -> None:
    volume = nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4))
    volume._affine = volume_affine
    mask = nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4))
    mask._affine = mask_affine

    with pytest.raises(ValueError, match=message):
        PreprocessInput(volume, mask, "QSM")


def test_preprocess_input_rejects_invalid_mask_spacing() -> None:
    volume = nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4))
    mask = nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4))
    mask.header.set_zooms((0.0, 1.0, 1.0))

    with pytest.raises(ValueError, match="Mask voxel spacing is invalid"):
        PreprocessInput(volume, mask, "QSM")
