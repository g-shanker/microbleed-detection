import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
from skimage.measure._regionprops import RegionProperties

from ...constants import COMPONENT_CONNECTIVITY, INVERTED_MODALITIES
from .. import io, utils
from ..datamodels import (
    FloatArray,
    IntArray,
    PreprocessInput,
    PreprocessOutput,
    Shape3D,
    VoxelSpacing,
)
from ..transforms import inpaint_vessels, volume_ops


def preprocess(
    preprocess_input: PreprocessInput,
) -> PreprocessOutput:
    """Canonicalize, normalize, crop, and inpaint an input brain volume."""
    volume = preprocess_input.volume
    mask = preprocess_input.mask
    modality = preprocess_input.modality
    canonical_volume = volume_ops.reorient_to_canonical(volume)

    if mask is not None:
        mask = volume_ops.reorient_to_canonical(mask)

    processed_volume = volume_ops.extract_brain(canonical_volume)
    if modality in INVERTED_MODALITIES:
        processed_volume = volume_ops.bias_field_correct_n4(processed_volume)

    volume_array = io.nifti_to_numpy(processed_volume)
    volume_array = volume_ops.normalize_volume(volume_array)

    if modality in INVERTED_MODALITIES:
        volume_array = volume_ops.invert_volume(volume_array)

    bounding_box = volume_ops.get_bounding_box(volume_array)
    volume_array = volume_ops.apply_bounding_box(volume_array, bounding_box)

    mask_array: IntArray | None = None
    if mask is not None:
        mask_array = io.nifti_to_numpy(mask).astype(np.uint8)
        mask_array = volume_ops.apply_bounding_box(mask_array, bounding_box)

    volume_array = inpaint_vessels.apply(volume_array)

    crop_start: Shape3D = (
        bounding_box[0][0],
        bounding_box[1][0],
        bounding_box[2][0],
    )
    canonical_affine: FloatArray = np.asarray(canonical_volume.affine, dtype=float)
    cropped_affine = volume_ops.adjust_affine_for_crop(canonical_affine, crop_start)

    return PreprocessOutput(
        volume=volume_array,
        mask=mask_array,
        affine=cropped_affine,
        bounding_box=bounding_box,
        original_volume=volume,
    )


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
    labels = utils.label_components(candidate_mask, COMPONENT_CONNECTIVITY)
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
    """Measure component elongation from its inertia eigenvalues."""
    eigenvalues = np.asarray(region.inertia_tensor_eigvals, dtype=float)
    maximum = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
    if maximum <= 0:
        return 0.0
    return 1.0 - float(np.min(eigenvalues)) / maximum
