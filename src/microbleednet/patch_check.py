import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TextIO

import numpy as np

from .core import io
from .core.datamodels import PatchRecord
from .orchestration.manifests import PatchManifest


@dataclass(frozen=True)
class PatchCheck:
    """Validation result for one related set of patch arrays."""

    volume_path: str
    mask_path: str
    frst_path: str
    record_count: int
    array_count: int
    shape: tuple[int, ...]
    volume_nonfinite_count: int
    mask_nonfinite_count: int
    frst_nonfinite_count: int
    volume_minimum: float | None
    volume_maximum: float | None
    frst_minimum: float | None
    frst_maximum: float | None
    issue: str


def finite_extrema(values: np.ndarray) -> tuple[float | None, float | None]:
    """Return extrema over finite values, or empty extrema when none exist."""
    finite_values = values[np.isfinite(values)]
    if not finite_values.size:
        return None, None
    return float(np.min(finite_values)), float(np.max(finite_values))


def shape_issues(
    volumes: np.ndarray,
    masks: np.ndarray,
    frst: np.ndarray,
    patch_size: int,
) -> list[str]:
    """Return dimensional and cross-array shape problems."""
    issues = []
    expected_tail = (patch_size, patch_size, patch_size)
    for name, values in (("volume", volumes), ("mask", masks), ("frst", frst)):
        if values.ndim != 4:
            issues.append(f"{name} array has {values.ndim} dimensions instead of 4")
        elif values.shape[1:] != expected_tail:
            issues.append(
                f"{name} patch shape is {values.shape[1:]}, expected {expected_tail}"
            )
    if volumes.shape != masks.shape or volumes.shape != frst.shape:
        issues.append(
            f"array shapes differ: volume={volumes.shape}, mask={masks.shape}, "
            f"frst={frst.shape}"
        )
    return issues


def record_issues(
    records: Sequence[PatchRecord], masks: np.ndarray, array_count: int
) -> list[str]:
    """Return manifest index, count, and classification-label problems."""
    issues = []
    indices = [record.patch_index for record in records]
    if len(set(indices)) != len(indices):
        issues.append("manifest contains duplicate patch indices")
    invalid_indices = [index for index in indices if index >= array_count]
    if invalid_indices:
        issues.append(f"manifest references out-of-range index {invalid_indices[0]}")
    if len(indices) != array_count:
        issues.append(
            f"manifest has {len(indices)} records for {array_count} stored patches"
        )
    if masks.ndim and masks.shape[0] >= array_count and not invalid_indices:
        mismatches = sum(
            bool(np.any(masks[record.patch_index] > 0)) != record.has_microbleed
            for record in records
        )
        if mismatches:
            issues.append(f"{mismatches} has_microbleed labels do not match masks")
    return issues


def inspect_group(
    records: Sequence[PatchRecord], patch_size: int
) -> PatchCheck:
    """Inspect one related set of volume, mask, and FRST patch arrays."""
    first = records[0]
    volumes = io.load_array_mmap(first.volume_path)
    masks = io.load_array_mmap(first.mask_path)
    frst = io.load_array_mmap(first.frst_path)
    issues = shape_issues(volumes, masks, frst, patch_size)

    volume_nonfinite_count = int(np.count_nonzero(~np.isfinite(volumes)))
    mask_nonfinite_count = int(np.count_nonzero(~np.isfinite(masks)))
    frst_nonfinite_count = int(np.count_nonzero(~np.isfinite(frst)))
    for name, count in (
        ("volume", volume_nonfinite_count),
        ("mask", mask_nonfinite_count),
        ("frst", frst_nonfinite_count),
    ):
        if count:
            issues.append(f"{name} contains {count} non-finite values")
    if np.any(np.isfinite(masks) & (masks != 0) & (masks != 1)):
        issues.append("mask contains values other than 0 and 1")

    array_count = int(volumes.shape[0]) if volumes.ndim else 0
    issues.extend(record_issues(records, masks, array_count))

    volume_minimum, volume_maximum = finite_extrema(volumes)
    frst_minimum, frst_maximum = finite_extrema(frst)
    return PatchCheck(
        volume_path=first.volume_path,
        mask_path=first.mask_path,
        frst_path=first.frst_path,
        record_count=len(records),
        array_count=array_count,
        shape=volumes.shape,
        volume_nonfinite_count=volume_nonfinite_count,
        mask_nonfinite_count=mask_nonfinite_count,
        frst_nonfinite_count=frst_nonfinite_count,
        volume_minimum=volume_minimum,
        volume_maximum=volume_maximum,
        frst_minimum=frst_minimum,
        frst_maximum=frst_maximum,
        issue="; ".join(issues),
    )


def inspect_manifest(manifest_path: Path) -> list[PatchCheck]:
    """Inspect every patch array and record referenced by a patch manifest."""
    manifest = PatchManifest.read(manifest_path)
    groups: dict[tuple[str, str, str], list[PatchRecord]] = defaultdict(list)
    for record in manifest.records:
        key = (record.volume_path, record.mask_path, record.frst_path)
        groups[key].append(record)
    return [
        inspect_group(records, manifest.patch_size) for records in groups.values()
    ]


def write_csv(checks: Sequence[PatchCheck], output: TextIO) -> None:
    """Write patch artifact checks as CSV."""
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "volume_path",
            "mask_path",
            "frst_path",
            "record_count",
            "array_count",
            "shape",
            "volume_nonfinite_count",
            "mask_nonfinite_count",
            "frst_nonfinite_count",
            "volume_minimum",
            "volume_maximum",
            "frst_minimum",
            "frst_maximum",
            "issue",
        )
    )
    for check in checks:
        writer.writerow(
            (
                check.volume_path,
                check.mask_path,
                check.frst_path,
                check.record_count,
                check.array_count,
                "x".join(str(size) for size in check.shape),
                check.volume_nonfinite_count,
                check.mask_nonfinite_count,
                check.frst_nonfinite_count,
                check.volume_minimum,
                check.volume_maximum,
                check.frst_minimum,
                check.frst_maximum,
                check.issue,
            )
        )


def main(arguments: Sequence[str] | None = None) -> None:
    """Validate a patch manifest and all referenced arrays."""
    parser = argparse.ArgumentParser(
        description="Check every volume, FRST, and mask array in a patch manifest."
    )
    parser.add_argument("manifest", type=Path, help="Path to a patch manifest")
    options = parser.parse_args(arguments)
    checks = inspect_manifest(options.manifest)
    write_csv(checks, sys.stdout)
    issue_count = sum(bool(check.issue) for check in checks)
    if issue_count:
        print(f"Found issues in {issue_count} patch array groups.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()