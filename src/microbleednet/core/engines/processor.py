import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
from skimage.measure._regionprops import RegionProperties

from .. import io, utils
from ..datamodels import (
    FloatArray,
    IntArray,
    Modality,
    PreprocessResult,
    Shape3D,
    VoxelSpacing,
)
from ..transforms import inpaint_vessels, volume_ops


def validate_volume_pair(
    volume: nib.Nifti1Image, mask: nib.Nifti1Image
) -> None:
    """Validate all image and mask invariants required by downstream stages."""
    if len(volume.shape) != 3:
        raise ValueError("image and mask must be 3D")
    if volume.shape != mask.shape:
        raise ValueError("image and mask shapes do not match")
    if volume.affine is None or mask.affine is None or not np.allclose(
        volume.affine, mask.affine
    ):
        raise ValueError("image and mask affines do not match")

    for name, image in (("image", volume), ("mask", mask)):
        spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(
            spacing <= 0
        ):
            raise ValueError(f"{name} voxel spacing must be positive and finite")

    volume_data = io.nifti_to_numpy(volume)
    mask_data = io.nifti_to_numpy(mask)
    if not np.isfinite(volume_data).all():
        raise ValueError("image contains non-finite values")
    if not np.isfinite(mask_data).all():
        raise ValueError("mask contains non-finite values")
    if np.max(volume_data) <= 0:
        raise ValueError("image is empty or its maximum is not positive")


def preprocess(
    volume: nib.Nifti1Image,
    mask: nib.Nifti1Image,
    modality: Modality,
) -> PreprocessResult:
    validate_volume_pair(volume, mask)
    canonical_volume = volume_ops.reorient_to_canonical(volume)

    mask = volume_ops.reorient_to_canonical(mask)

    processed_volume = volume_ops.extract_brain(canonical_volume)
    if modality in {"T2*-GRE", "SWI"}:
        processed_volume = volume_ops.bias_field_correct_n4(processed_volume)

    volume_array = io.nifti_to_numpy(processed_volume)
    volume_array = volume_ops.normalize_volume(volume_array)

    if modality in {"T2*-GRE", "SWI"}:
        volume_array = volume_ops.invert_volume(volume_array)

    volume_array, bounding_box = volume_ops.tight_crop_volume(volume_array)
    mask_array: IntArray = io.nifti_to_numpy(mask).astype(np.uint8)
    mask_array = volume_ops.apply_bounding_box(mask_array, bounding_box)

    volume_array = inpaint_vessels.apply(volume_array)

    crop_start: Shape3D = (
        bounding_box[0][0],
        bounding_box[1][0],
        bounding_box[2][0],
    )
    canonical_affine: FloatArray = np.asarray(canonical_volume.affine, dtype=float)
    cropped_affine = volume_ops.adjust_affine_for_crop(canonical_affine, crop_start)

    return PreprocessResult(volume_array, mask_array, cropped_affine)


def postprocess(
    candidate_mask: np.ndarray,
    volume: np.ndarray,
    voxel_sizes: VoxelSpacing,
    minimum_volume_mm3: float,
    maximum_ellipticity: float,
    minimum_brain_distance_mm: float,
) -> np.ndarray:
    """Apply volume, shape, and brain-boundary filters."""
    brain_mask = volume > 0
    brain_distance = np.asarray(
        distance_transform_edt(brain_mask, sampling=voxel_sizes)
    )
    labels = utils.label_components(candidate_mask, utils.COMPONENT_CONNECTIVITY)
    output = np.zeros_like(candidate_mask, dtype=np.uint8)
    voxel_volume = float(np.prod(voxel_sizes))
    voxel_regions = regionprops(labels)
    physical_regions = regionprops(labels, spacing=voxel_sizes)
    for voxel_region, physical_region in zip(
        voxel_regions, physical_regions, strict=True
    ):
        if voxel_region.area * voxel_volume < minimum_volume_mm3:
            continue
        if component_ellipticity(physical_region) > maximum_ellipticity:
            continue
        centroid: Shape3D = (
            int(round(voxel_region.centroid[0])),
            int(round(voxel_region.centroid[1])),
            int(round(voxel_region.centroid[2])),
        )
        if brain_distance[centroid] < minimum_brain_distance_mm:
            continue
        output[labels == voxel_region.label] = 1
    return output


def component_ellipticity(region: RegionProperties) -> float:
    eigenvalues = np.asarray(region.inertia_tensor_eigvals, dtype=float)
    maximum = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
    if maximum <= 0:
        return 0.0
    return 1.0 - float(np.min(eigenvalues)) / maximum
