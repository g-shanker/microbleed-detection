import os
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

from ...errors import ApplicationError
from .. import io
from ..datamodels import BoundingBox, Shape3D


def translate(volume: np.ndarray, offset_x: int, offset_y: int) -> np.ndarray:
    """Shift the first two axes without wrapping values across boundaries."""
    translated = np.zeros_like(volume)
    source_x_start = max(0, -offset_x)
    source_x_stop = min(volume.shape[0], volume.shape[0] - offset_x)
    source_y_start = max(0, -offset_y)
    source_y_stop = min(volume.shape[1], volume.shape[1] - offset_y)
    target_x_start = max(0, offset_x)
    target_y_start = max(0, offset_y)
    translated[
        target_x_start : target_x_start + source_x_stop - source_x_start,
        target_y_start : target_y_start + source_y_stop - source_y_start,
        ...,
    ] = volume[source_x_start:source_x_stop, source_y_start:source_y_stop, ...]
    return translated


def add_noise(volume: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Add a precomputed noise array to a volume."""
    if noise.shape != volume.shape:
        raise ApplicationError(
            category="Input data",
            summary="Noise and volume shapes do not match",
            cause=f"Volume shape is {volume.shape}, noise shape is {noise.shape}",
            fix="Generate noise with the target volume shape",
        )
    return volume + noise


def blur(volume: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian filtering with the supplied sigma."""
    return gaussian_filter(volume, sigma)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    maximum = np.max(volume)
    if not np.isfinite(maximum) or maximum <= 0:
        raise ApplicationError(
            category="Input data",
            summary="Volume cannot be normalized",
            cause="Normalization requires a finite positive maximum",
            fix="Verify that the input volume is non-empty and correctly loaded",
        )
    return volume / maximum


def invert_volume(volume: np.ndarray) -> np.ndarray:
    brain_mask = volume > 0
    volume = np.max(volume) - volume
    volume = volume * brain_mask
    return volume


def get_bounding_box(volume: np.ndarray) -> BoundingBox:
    positive_voxels = np.argwhere(volume > 0)
    if positive_voxels.size == 0:
        raise ApplicationError(
            category="Input data",
            summary="Cannot compute a bounding box without positive voxels",
            fix="Verify the mask or volume before cropping",
        )

    starts = positive_voxels.min(axis=0)
    stops = positive_voxels.max(axis=0) + 1
    return (
        (int(starts[0]), int(stops[0])),
        (int(starts[1]), int(stops[1])),
        (int(starts[2]), int(stops[2])),
    )


def apply_bounding_box(volume: np.ndarray, bounding_box: BoundingBox) -> np.ndarray:
    (d0_start, d0_end), (d1_start, d1_end), (d2_start, d2_end) = bounding_box
    cropped_volume = volume[d0_start:d0_end, d1_start:d1_end, d2_start:d2_end]
    return cropped_volume


def reorient_to_canonical(volume: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.as_closest_canonical(volume)


def adjust_affine_for_crop(
    affine: np.ndarray,
    crop_start: Shape3D,
) -> np.ndarray:
    translation = np.eye(4)
    translation[:3, 3] = crop_start
    return affine @ translation


def restore_cropped_volume(
    cropped_volume: np.ndarray,
    bounding_box: BoundingBox,
    original_volume: nib.Nifti1Image,
) -> nib.Nifti1Image:
    """Restore a canonical cropped array to the original image orientation."""
    canonical_volume = nib.as_closest_canonical(original_volume)
    restored_canonical = np.zeros(canonical_volume.shape, dtype=cropped_volume.dtype)
    (d0_start, d0_end), (d1_start, d1_end), (d2_start, d2_end) = bounding_box
    restored_canonical[d0_start:d0_end, d1_start:d1_end, d2_start:d2_end] = (
        cropped_volume
    )
    transform = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(canonical_volume.affine),
        nib.orientations.io_orientation(original_volume.affine),
    )
    restored = nib.orientations.apply_orientation(restored_canonical, transform)
    return nib.Nifti1Image(
        restored.astype(np.uint8),
        original_volume.affine,
        original_volume.header,
    )


def extract_brain(volume: nib.Nifti1Image) -> nib.Nifti1Image:
    fsldir = Path(os.getenv("FSLDIR", ""))
    if not fsldir.is_dir():
        raise ApplicationError(
            category="Environment",
            summary="Brain extraction requires FSL",
            cause="FSLDIR is missing or is not a directory",
            fix="Install FSL and set FSLDIR to its installation directory",
        )

    with tempfile.TemporaryDirectory(prefix="microbleednet-fsl-bet-") as temp_dir:
        temp_dir = Path(temp_dir)
        input_path = temp_dir / "pre_bet.nii.gz"
        output_path = temp_dir / "post_bet.nii.gz"

        # Save to disk just for BET
        io.save_volume(volume, input_path)
        subprocess.run(
            [str(fsldir / "bin" / "bet"), str(input_path), str(output_path)], check=True
        )

        # Some potentially over-defensive programming:
        # Load back into memory immediately and let tempdir delete the files
        output_volume = io.load_volume(output_path)
        # Force load data into memory so we don't rely on the deleted temp file
        output_data = io.nifti_to_numpy(output_volume)
        loaded_volume = io.numpy_to_nifti(output_data, output_volume)

        return loaded_volume


def bias_field_correct_n4(volume: nib.Nifti1Image) -> nib.Nifti1Image:
    volume_data = io.nifti_to_numpy(volume)

    sitk_volume = sitk.GetImageFromArray(volume_data.T)
    mask_image = sitk_volume > 0
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_sitk = corrector.Execute(sitk_volume, mask_image)
    corrected_data = sitk.GetArrayFromImage(corrected_sitk).T
    corrected_nifti = io.numpy_to_nifti(corrected_data, volume)

    return corrected_nifti
