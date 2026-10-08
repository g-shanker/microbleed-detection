import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TextIO

import numpy as np

from .core import io
from .orchestration.manifests import PreprocessedDatasetManifest


@dataclass(frozen=True)
class VolumeCheck:
    """Validation result for one persisted preprocessing artifact."""

    subject_id: str
    variant_index: int | None
    artifact: str
    path: str
    shape: tuple[int, ...]
    minimum: float | None
    maximum: float | None
    nonfinite_count: int
    negative_count: int
    above_one_count: int
    issue: str


def inspect_volume(
    subject_id: str,
    variant_index: int | None,
    artifact: str,
    path: str,
    expected_shape: tuple[int, ...] | None,
    binary: bool = False,
) -> VolumeCheck:
    """Inspect one NIfTI artifact for shape, range, and finite values."""
    values = io.nifti_to_numpy(io.load_volume(path))
    finite = np.isfinite(values)
    finite_values = values[finite]
    issues = []
    if values.ndim != 3:
        issues.append(f"expected 3 dimensions, found {values.ndim}")
    if expected_shape is not None and values.shape != expected_shape:
        issues.append(f"expected shape {expected_shape}, found {values.shape}")
    nonfinite_count = int(np.count_nonzero(~finite))
    if nonfinite_count:
        issues.append(f"contains {nonfinite_count} non-finite values")
    if binary and np.any(finite & (values != 0) & (values != 1)):
        issues.append("contains values other than 0 and 1")

    return VolumeCheck(
        subject_id=subject_id,
        variant_index=variant_index,
        artifact=artifact,
        path=path,
        shape=values.shape,
        minimum=float(np.min(finite_values)) if finite_values.size else None,
        maximum=float(np.max(finite_values)) if finite_values.size else None,
        nonfinite_count=nonfinite_count,
        negative_count=int(np.count_nonzero(finite & (values < 0))),
        above_one_count=int(np.count_nonzero(finite & (values > 1))),
        issue="; ".join(issues),
    )


def inspect_manifest(manifest_path: Path) -> list[VolumeCheck]:
    """Inspect every persisted artifact in a preprocessed dataset manifest."""
    manifest = PreprocessedDatasetManifest.read(manifest_path)
    checks = []
    for subject in manifest.subjects:
        brain_check = inspect_volume(
            subject.subject_id,
            None,
            "brain_mask",
            subject.brain_mask_path,
            None,
            binary=True,
        )
        checks.append(brain_check)
        expected_shape = brain_check.shape
        for variant_index, variant in enumerate(subject.variants):
            checks.append(
                inspect_volume(
                    subject.subject_id,
                    variant_index,
                    "volume",
                    variant.volume_path,
                    expected_shape,
                )
            )
            checks.append(
                inspect_volume(
                    subject.subject_id,
                    variant_index,
                    "frst",
                    variant.frst_path,
                    expected_shape,
                )
            )
            if variant.mask_path is not None:
                checks.append(
                    inspect_volume(
                        subject.subject_id,
                        variant_index,
                        "lesion_mask",
                        variant.mask_path,
                        expected_shape,
                        binary=True,
                    )
                )
    return checks


def write_csv(checks: Sequence[VolumeCheck], output: TextIO) -> None:
    """Write preprocessing artifact checks as CSV."""
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "subject_id",
            "variant_index",
            "artifact",
            "path",
            "shape",
            "minimum",
            "maximum",
            "nonfinite_count",
            "negative_count",
            "above_one_count",
            "issue",
        )
    )
    for check in checks:
        writer.writerow(
            (
                check.subject_id,
                check.variant_index,
                check.artifact,
                check.path,
                "x".join(str(size) for size in check.shape),
                check.minimum,
                check.maximum,
                check.nonfinite_count,
                check.negative_count,
                check.above_one_count,
                check.issue,
            )
        )


def main(arguments: Sequence[str] | None = None) -> None:
    """Validate a preprocessed dataset manifest and its artifacts."""
    parser = argparse.ArgumentParser(
        description="Check every volume, FRST result, and mask in preprocessed.json."
    )
    parser.add_argument("manifest", type=Path, help="Path to preprocessed.json")
    options = parser.parse_args(arguments)
    checks = inspect_manifest(options.manifest)
    write_csv(checks, sys.stdout)
    issue_count = sum(bool(check.issue) for check in checks)
    if issue_count:
        print(f"Found issues in {issue_count} artifacts.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()