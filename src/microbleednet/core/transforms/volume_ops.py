import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from ...errors import ApplicationError
from .. import io
from ..datamodels import BoundingBox, Shape3D
from ..utils import fsl_executable, run_fsl


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
    """Scale volume intensities by their finite positive maximum."""
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
    """Invert positive brain intensities while preserving the zero background."""
    brain_mask = volume > 0
    volume = np.max(volume) - volume
    volume = volume * brain_mask
    return volume


def get_bounding_box(volume: np.ndarray) -> BoundingBox:
    """Return the minimal exclusive-stop bounds containing all positive voxels."""
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
    """Crop a volume to the supplied three-dimensional bounds."""
    (d0_start, d0_end), (d1_start, d1_end), (d2_start, d2_end) = bounding_box
    cropped_volume = volume[d0_start:d0_end, d1_start:d1_end, d2_start:d2_end]
    return cropped_volume


def reorient_to_canonical(volume: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient a NIfTI image to the closest canonical voxel orientation."""
    return nib.as_closest_canonical(volume)


def adjust_affine_for_crop(
    affine: np.ndarray,
    crop_start: Shape3D,
) -> np.ndarray:
    """Translate an affine origin to account for a crop's starting voxel."""
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


def extract_brain(
    volume: nib.Nifti1Image,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Extract brain tissue and its binary mask with FSL BET."""
    bet_path = fsl_executable("bet")

    with tempfile.TemporaryDirectory(prefix="microbleednet-fsl-bet-") as temp_dir:
        temp_dir = Path(temp_dir)
        input_path = temp_dir / "pre_bet.nii.gz"
        output_path = temp_dir / "post_bet.nii.gz"
        mask_path = temp_dir / "post_bet_mask.nii.gz"

        # Save to disk just for BET
        io.save_volume(volume, input_path)
        run_fsl(
            [str(bet_path), str(input_path), str(output_path), "-m"],
            "BET",
        )

        # Some potentially over-defensive programming:
        # Load back into memory immediately and let tempdir delete the files
        output_volume = io.load_volume(output_path)
        output_data = io.nifti_to_numpy(output_volume)
        loaded_volume = io.numpy_to_nifti(output_data, output_volume)
        mask_volume = io.load_volume(mask_path)
        mask_data = io.nifti_to_numpy(mask_volume)
        loaded_mask = io.numpy_to_nifti(mask_data, mask_volume)

        return loaded_volume, loaded_mask


def bias_field_correct_fast(volume: nib.Nifti1Image) -> nib.Nifti1Image:
    """Correct intensity bias with FSL FAST and materialize its restored image."""
    fast_path = fsl_executable("fast")
    with tempfile.TemporaryDirectory(prefix="microbleednet-fsl-fast-") as temp_dir:
        temp_dir = Path(temp_dir)
        input_path = temp_dir / "pre_fast.nii.gz"
        output_prefix = temp_dir / "fast"
        restored_path = temp_dir / "fast_restore.nii.gz"
        io.save_volume(volume, input_path)
        run_fsl(
            [str(fast_path), "-B", "-o", str(output_prefix), str(input_path)],
            "FAST",
        )
        corrected_volume = io.load_volume(restored_path)
        corrected_data = io.nifti_to_numpy(corrected_volume)
        return io.numpy_to_nifti(corrected_data, corrected_volume)
