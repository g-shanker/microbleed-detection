import argparse
import csv
import math
import sys
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Sequence, TextIO

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt


from .progress import progress

from .constants import (
    COMPONENT_CONNECTIVITY,
    FRST_RADII,
    INVERTED_MODALITIES,
    VARIANT_INDEX,
)
from .core import io
from .core.transforms import patch, volume_ops
from .core.utils import label_components
from .errors import ApplicationError
from .orchestration.layouts import DatasetLayout, ExperimentLayout
from .orchestration.manifests import (
    InferManifest,
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    RawDatasetManifest,
)

DEFAULT_SEARCH_RADIUS_MM = 3.0
CORE_RADIUS_MM = 1.5
RING_INNER_MM = 3.0
RING_OUTER_MM = 6.0
MINIMUM_RING_VOXELS = 10
MAD_TO_STD = 1.4826
STRONG_FRST_FRACTION = 0.5
CANONICAL_SLICE_AXIS = 2
INPUT_DATA_CATEGORY = "Input data"
SUMMARY_COLUMNS = (
    "annotation_voxels",
    "nearest_detection_mm",
    "probability_at_point",
    "max_probability_nearby",
    "brain_edge_distance_mm",
    "raw_contrast",
    "preprocessed_contrast",
    "frst_contrast",
    "frst_slice_fraction",
    "slice_peak_edge_distance_mm",
    "slice_strong_frst_near_edge",
)


@dataclass(frozen=True)
class LesionRecord:
    """Detection outcome and local image evidence for one annotated lesion."""

    subject_id: str
    lesion: int
    voxel_i: int
    voxel_j: int
    voxel_k: int
    world_x_mm: float
    world_y_mm: float
    world_z_mm: float
    annotation_voxels: int
    detected: bool
    nearest_detection_mm: float
    probability_at_point: float
    max_probability_nearby: float
    max_probability_offset_mm: float
    in_brain_mask: bool
    brain_edge_distance_mm: float
    raw_contrast: float
    preprocessed_contrast: float
    frst_contrast: float
    frst_slice_fraction: float
    slice_peak_edge_distance_mm: float
    slice_strong_frst_near_edge: float


@dataclass(frozen=True)
class SubjectArrays:
    """Raw-space arrays needed to describe each annotated lesion."""

    reference: np.ndarray
    raw_volume: np.ndarray
    probability: np.ndarray
    detections: np.ndarray
    brain_mask: np.ndarray
    preprocessed_volume: np.ndarray
    frst: np.ndarray
    affine: np.ndarray
    spacing: np.ndarray
    raw_lesions_bright: bool
    frst_slice_axis: int


def inspect_run(
    dataset_dir: Path,
    run_dir: Path,
    search_radius_mm: float = DEFAULT_SEARCH_RADIUS_MM,
) -> list[LesionRecord]:
    """Describe every annotated lesion for the subjects of an inference run."""
    dataset_layout = DatasetLayout(dataset_dir=dataset_dir)
    inference_manifest_path = ExperimentLayout(
        experiment_dir=run_dir
    ).inference_manifest_path()
    inference_manifest = InferManifest.read(inference_manifest_path)
    if inference_manifest.status is not ManifestStatus.COMPLETE:
        raise ApplicationError(
            category="Manifest",
            summary="Cannot inspect incomplete inference output",
            cause=(
                f"Inference manifest status is '{inference_manifest.status.value}'"
            ),
            fix="Rerun evaluation or inference to completion",
            context={"path": str(inference_manifest_path)},
        )
    raw_manifest = RawDatasetManifest.read(dataset_layout.raw_manifest_path())
    preprocessed_manifest = PreprocessedDatasetManifest.read(
        dataset_layout.preprocessed_manifest_path()
    )
    raw_subjects = {subject.subject_id: subject for subject in raw_manifest.subjects}
    preprocessed_subjects = {
        subject.subject_id: subject for subject in preprocessed_manifest.subjects
    }
    modalities = {
        source.source_id: source.modality for source in raw_manifest.sources
    }

    records: list[LesionRecord] = []
    for inferred in progress.track(inference_manifest.subjects):
        subject_id = inferred.subject_id
        raw_subject = raw_subjects.get(subject_id)
        preprocessed_subject = preprocessed_subjects.get(subject_id)
        if raw_subject is None or preprocessed_subject is None:
            raise ApplicationError(
                category="Manifest",
                summary="Inferred subject is missing from the dataset manifests",
                cause=(
                    f"Subject '{subject_id}' is not in both raw.json and "
                    "preprocessed.json"
                ),
                fix="Pass the dataset directory that produced this inference run",
                context={"subject_id": subject_id},
            )
        if raw_subject.mask_path is None:
            raise ApplicationError(
                category=INPUT_DATA_CATEGORY,
                summary="Lesion inspection requires a reference mask",
                cause=f"Subject '{subject_id}' has no reference mask",
                fix="Inspect only runs whose subjects have reference masks",
                context={"subject_id": subject_id},
            )
        raw_image = io.load_volume(raw_subject.volume_path)
        arrays = SubjectArrays(
            reference=io.nifti_to_numpy(io.load_volume(raw_subject.mask_path)),
            raw_volume=io.nifti_to_numpy(raw_image),
            probability=io.nifti_to_numpy(
                io.load_volume(inferred.detector_output_path)
            ),
            detections=io.nifti_to_numpy(
                io.load_volume(inferred.discriminator_output_path)
            ),
            brain_mask=restore_preprocessed(
                preprocessed_subject.brain_mask_path, preprocessed_subject
            ),
            preprocessed_volume=restore_preprocessed(
                preprocessed_subject.variants[VARIANT_INDEX].volume_path,
                preprocessed_subject,
            ),
            frst=restore_preprocessed(
                preprocessed_subject.variants[VARIANT_INDEX].frst_path,
                preprocessed_subject,
            ),
            affine=np.asarray(raw_image.affine, dtype=float),
            spacing=np.asarray(raw_image.header.get_zooms()[:3], dtype=float),
            raw_lesions_bright=(
                modalities[raw_subject.source_id] not in INVERTED_MODALITIES
            ),
            frst_slice_axis=frst_slice_axis(raw_image.affine),
        )
        records.extend(inspect_subject(subject_id, arrays, search_radius_mm))
    return records


def restore_preprocessed(path: str, subject: PreprocessedSubject) -> np.ndarray:
    """Map a cropped canonical preprocessed array back to raw voxel space."""
    restored = volume_ops.restore_cropped_volume(
        io.nifti_to_numpy(io.load_volume(path)),
        subject.bounding_box,
        io.load_volume(subject.original_volume_path),
    )
    return io.nifti_to_numpy(restored)


def inspect_subject(
    subject_id: str, arrays: SubjectArrays, search_radius_mm: float
) -> list[LesionRecord]:
    """Describe each connected reference component in one subject."""
    shapes = {
        name: getattr(arrays, name).shape
        for name in (
            "reference",
            "raw_volume",
            "probability",
            "detections",
            "brain_mask",
            "preprocessed_volume",
            "frst",
        )
    }
    if len(set(shapes.values())) != 1:
        raise ApplicationError(
            category=INPUT_DATA_CATEGORY,
            summary="Raw-space arrays do not share one shape",
            cause=f"Subject '{subject_id}' has shapes {shapes}",
            fix="Check that the run and dataset directories belong together",
            context={"subject_id": subject_id},
        )

    labels = label_components(arrays.reference > 0, COMPONENT_CONNECTIVITY)
    detections = arrays.detections > 0
    brain = arrays.brain_mask > 0
    detection_distance = (
        np.asarray(distance_transform_edt(~detections, sampling=arrays.spacing))
        if detections.any()
        else np.full(detections.shape, np.nan)
    )
    brain_distance = np.asarray(
        distance_transform_edt(brain, sampling=arrays.spacing)
    )

    records = []
    for lesion, center in enumerate(patch.get_target_centers(labels), start=1):
        component = labels == labels[center]
        world = arrays.affine @ np.array([*center, 1.0])
        nearby_max, nearby_offset = max_within_radius(
            arrays.probability, center, arrays.spacing, search_radius_mm
        )
        slice_fraction, peak_edge_distance, strong_near_edge = frst_slice_evidence(
            arrays.frst,
            center,
            arrays.frst_slice_axis,
            brain_distance,
            arrays.spacing,
        )
        records.append(
            LesionRecord(
                subject_id=subject_id,
                lesion=lesion,
                voxel_i=center[0],
                voxel_j=center[1],
                voxel_k=center[2],
                world_x_mm=float(world[0]),
                world_y_mm=float(world[1]),
                world_z_mm=float(world[2]),
                annotation_voxels=int(component.sum()),
                detected=bool(detections[component].any()),
                nearest_detection_mm=float(detection_distance[center]),
                probability_at_point=float(arrays.probability[center]),
                max_probability_nearby=nearby_max,
                max_probability_offset_mm=nearby_offset,
                in_brain_mask=bool(brain[center]),
                brain_edge_distance_mm=float(brain_distance[center]),
                raw_contrast=local_contrast(
                    arrays.raw_volume,
                    center,
                    arrays.spacing,
                    brain,
                    arrays.raw_lesions_bright,
                ),
                preprocessed_contrast=local_contrast(
                    arrays.preprocessed_volume, center, arrays.spacing, brain, True
                ),
                frst_contrast=local_contrast(
                    arrays.frst, center, arrays.spacing, brain, True
                ),
                frst_slice_fraction=slice_fraction,
                slice_peak_edge_distance_mm=peak_edge_distance,
                slice_strong_frst_near_edge=strong_near_edge,
            )
        )
    return records


def frst_slice_axis(affine: np.ndarray) -> int:
    """Return the raw axis that maps to the canonical axis FRST slices along."""
    orientation = nib.orientations.io_orientation(affine)
    return int(np.flatnonzero(orientation[:, 0] == CANONICAL_SLICE_AXIS)[0])


def frst_slice_evidence(
    frst: np.ndarray,
    center: tuple[int, int, int],
    axis: int,
    brain_distance: np.ndarray,
    spacing: np.ndarray,
) -> tuple[float, float, float]:
    """Compare a lesion's FRST with the strongest response in its FRST slice."""
    # FRST is normalized per slice, so a stronger structure elsewhere in the
    # slice suppresses the lesion's response.
    plane_index = tuple(
        center[dimension] if dimension == axis else slice(None)
        for dimension in range(3)
    )
    plane = frst[plane_index]
    peak = float(plane.max())
    if peak <= 0:
        return math.nan, math.nan, math.nan

    window, distance = local_distances(frst.shape, center, spacing, CORE_RADIUS_MM)
    offset = center[axis] - int(window[axis].start)
    core_values = np.take(frst[window], offset, axis=axis)
    core_distance = np.take(distance, offset, axis=axis)
    core = float(core_values[core_distance <= CORE_RADIUS_MM].max())

    plane_edge = brain_distance[plane_index]
    peak_edge_distance = float(plane_edge.flat[int(np.argmax(plane))])
    # FRST votes from the brain boundary land up to its largest radius inside.
    edge_band_mm = (max(FRST_RADII) + 1) * float(np.delete(spacing, axis).max())
    strong = plane >= STRONG_FRST_FRACTION * peak
    strong_near_edge = float(np.mean(plane_edge[strong] <= edge_band_mm))
    return core / peak, peak_edge_distance, strong_near_edge


def local_distances(
    shape: tuple[int, ...],
    center: tuple[int, int, int],
    spacing: np.ndarray,
    radius_mm: float,
) -> tuple[tuple[slice, ...], np.ndarray]:
    """Return a clipped window around a voxel and its distances in mm."""
    half_extent = np.ceil(radius_mm / spacing).astype(int)
    window = tuple(
        slice(max(c - h, 0), min(c + h + 1, size))
        for c, h, size in zip(center, half_extent, shape)
    )
    grid = np.ogrid[window]
    distance = np.sqrt(
        sum(((axis - c) * s) ** 2 for axis, c, s in zip(grid, center, spacing))
    )
    return window, np.asarray(distance)


def max_within_radius(
    probability: np.ndarray,
    center: tuple[int, int, int],
    spacing: np.ndarray,
    radius_mm: float,
) -> tuple[float, float]:
    """Return the highest probability near a point and its distance in mm."""
    window, distance = local_distances(probability.shape, center, spacing, radius_mm)
    values = np.where(distance <= radius_mm, probability[window], -np.inf)
    index = np.unravel_index(int(np.argmax(values)), values.shape)
    return float(values[index]), float(distance[index])


def local_contrast(
    volume: np.ndarray,
    center: tuple[int, int, int],
    spacing: np.ndarray,
    brain: np.ndarray,
    lesions_bright: bool,
) -> float:
    """Score how far a lesion core stands out from its surrounding brain."""
    window, distance = local_distances(volume.shape, center, spacing, RING_OUTER_MM)
    values = volume[window] if lesions_bright else -volume[window]
    ring = (distance >= RING_INNER_MM) & (distance <= RING_OUTER_MM) & brain[window]
    if int(ring.sum()) < MINIMUM_RING_VOXELS:
        return math.nan
    background = values[ring]
    median = float(np.median(background))
    spread = MAD_TO_STD * float(np.median(np.abs(background - median)))
    if spread == 0:
        return math.nan
    core = float(values[distance <= CORE_RADIUS_MM].max())
    return (core - median) / spread


def write_csv(records: Sequence[LesionRecord], output: TextIO) -> None:
    """Write one CSV row per annotated lesion."""
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(field.name for field in fields(LesionRecord))
    for record in records:
        writer.writerow(
            round(value, 4) if isinstance(value, float) else value
            for value in astuple(record)
        )


def write_summary(records: Sequence[LesionRecord], output: TextIO) -> None:
    """Compare median lesion properties between detected and missed lesions."""
    groups = {
        "detected": [record for record in records if record.detected],
        "missed": [record for record in records if not record.detected],
    }
    output.write(
        f"lesions={len(records)} detected={len(groups['detected'])} "
        f"missed={len(groups['missed'])}\n"
    )
    output.write(f"{'median':<24}{'detected':>12}{'missed':>12}\n")
    for column in SUMMARY_COLUMNS:
        medians = [median_of(group, column) for group in groups.values()]
        output.write(
            f"{column:<24}" + "".join(f"{value:>12.3f}" for value in medians) + "\n"
        )
    for name, group in groups.items():
        outside = sum(not record.in_brain_mask for record in group)
        output.write(f"{name} outside brain mask: {outside}/{len(group)}\n")


def median_of(records: Sequence[LesionRecord], column: str) -> float:
    """Return the median of the finite values of one numeric column."""
    values = [float(getattr(record, column)) for record in records]
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else math.nan


def main(arguments: Sequence[str] | None = None) -> None:
    """Run the per-lesion detection report."""
    parser = argparse.ArgumentParser(
        description=(
            "Report detector evidence and image contrast for every annotated "
            "lesion in an evaluation or inference run."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Dataset directory containing manifests/raw.json and preprocessed.json",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Evaluation output or experiment directory with manifests/infer.json",
    )
    parser.add_argument(
        "--search-radius-mm",
        type=float,
        default=DEFAULT_SEARCH_RADIUS_MM,
        help="Radius for the nearby maximum detector probability",
    )
    options = parser.parse_args(arguments)
    records = inspect_run(
        options.dataset_dir, options.run_dir, options.search_radius_mm
    )
    write_csv(records, sys.stdout)
    write_summary(records, sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()
