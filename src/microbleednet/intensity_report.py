import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TextIO

import numpy as np

from .core import io
from .errors import ApplicationError
from .orchestration.manifests import PreprocessedDatasetManifest

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
INPUT_DATA_CATEGORY = "Input data"


@dataclass(frozen=True)
class IntensityStatistics:
    """Brain-masked intensity statistics for one subject's original variant."""

    subject_id: str
    voxel_count: int
    minimum: float
    percentiles: tuple[float, ...]
    maximum: float
    mean: float
    standard_deviation: float
    zero_fraction: float
    negative_fraction: float
    above_one_fraction: float
    nonfinite_fraction: float


def inspect_manifest(manifest_path: Path) -> list[IntensityStatistics]:
    """Inspect variant zero for every subject in a preprocessed manifest."""
    manifest = PreprocessedDatasetManifest.read(manifest_path)
    statistics = []
    for subject in manifest.subjects:
        if not subject.variants:
            raise ApplicationError(
                category="Manifest",
                summary="Preprocessed subject has no variants",
                cause=f"Subject '{subject.subject_id}' has no variant 0",
                fix="Rerun preprocessing for the subject",
                context={"path": str(manifest_path)},
            )

        volume = io.nifti_to_numpy(io.load_volume(subject.variants[0].volume_path))
        brain_mask = io.nifti_to_numpy(io.load_volume(subject.brain_mask_path)) > 0
        if volume.shape != brain_mask.shape:
            raise ApplicationError(
                category=INPUT_DATA_CATEGORY,
                summary="Variant volume and brain mask shapes do not match",
                cause=(
                    f"Subject '{subject.subject_id}' has volume shape {volume.shape} "
                    f"and brain mask shape {brain_mask.shape}"
                ),
                fix="Rerun preprocessing for the subject",
                context={"subject_id": subject.subject_id},
            )

        brain_values = volume[brain_mask]
        if brain_values.size == 0:
            raise ApplicationError(
                category=INPUT_DATA_CATEGORY,
                summary="Brain mask contains no foreground voxels",
                cause=f"Subject '{subject.subject_id}' has an empty brain mask",
                fix="Verify the brain extraction output and rerun preprocessing",
                context={"subject_id": subject.subject_id},
            )

        finite = np.isfinite(brain_values)
        finite_values = brain_values[finite]
        if finite_values.size == 0:
            raise ApplicationError(
                category=INPUT_DATA_CATEGORY,
                summary="Variant contains no finite brain intensities",
                cause=f"Subject '{subject.subject_id}' has no finite values",
                fix="Verify the source volume and rerun preprocessing",
                context={"subject_id": subject.subject_id},
            )

        statistics.append(
            IntensityStatistics(
                subject_id=subject.subject_id,
                voxel_count=int(brain_values.size),
                minimum=float(np.min(finite_values)),
                percentiles=tuple(
                    float(value)
                    for value in np.percentile(finite_values, PERCENTILES)
                ),
                maximum=float(np.max(finite_values)),
                mean=float(np.mean(finite_values)),
                standard_deviation=float(np.std(finite_values)),
                zero_fraction=float(np.mean(finite_values == 0)),
                negative_fraction=float(np.mean(finite_values < 0)),
                above_one_fraction=float(np.mean(finite_values > 1)),
                nonfinite_fraction=float(np.mean(~finite)),
            )
        )
    return statistics


def write_csv(statistics: Sequence[IntensityStatistics], output: TextIO) -> None:
    """Write subject statistics as CSV."""
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "subject_id",
            "brain_voxels",
            "minimum",
            *(f"p{percentile:02d}" for percentile in PERCENTILES),
            "maximum",
            "mean",
            "standard_deviation",
            "zero_fraction",
            "negative_fraction",
            "above_one_fraction",
            "nonfinite_fraction",
        )
    )
    for subject in statistics:
        writer.writerow(
            (
                subject.subject_id,
                subject.voxel_count,
                subject.minimum,
                *subject.percentiles,
                subject.maximum,
                subject.mean,
                subject.standard_deviation,
                subject.zero_fraction,
                subject.negative_fraction,
                subject.above_one_fraction,
                subject.nonfinite_fraction,
            )
        )


def main(arguments: Sequence[str] | None = None) -> None:
    """Run the variant-zero intensity report."""
    parser = argparse.ArgumentParser(
        description=(
            "Report brain-masked intensity statistics for variant 0 of every "
            "subject in a preprocessed manifest."
        )
    )
    parser.add_argument("manifest", type=Path, help="Path to preprocessed.json")
    options = parser.parse_args(arguments)
    write_csv(inspect_manifest(options.manifest), sys.stdout)


if __name__ == "__main__":  # pragma: no cover
    main()