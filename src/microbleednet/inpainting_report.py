"""Choose the vessel-mask dilation used before inpainting.

Experimental diagnostic. See EXPERIMENTAL_CHANGES.md for how to remove it.
"""

import argparse
import csv
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence, TextIO, cast

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    generate_binary_structure,
)
from skimage.filters import frangi

from . import report_checkpoints
from .constants import (
    COMPONENT_CONNECTIVITY,
    FRANGI_ALPHA,
    FRANGI_BETA,
    FRANGI_BLACK_RIDGES,
    FRANGI_SIGMAS,
)
from .core import io
from .core.datamodels import BiasFieldCorrection, PreprocessInput, PreprocessOutput
from .core.engines import processor
from .core.transforms import frst, inpaint_vessels, patch
from .core.utils import label_components
from .errors import ApplicationError
from .lesion_report import local_contrast
from .orchestration.layouts import DatasetLayout
from .orchestration.manifests import RawDatasetManifest
from .orchestration.utils import subject_failure_context

logger = logging.getLogger(__name__)

DILATION_MODES = ("3d", "in_plane")
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_FRST_THRESHOLD = 0.2
LESION_EXCLUSION_MM = 3.0
REDUCTION_TARGET = 0.9
MAXIMUM_EXTRA_TOUCHED = 0.02
CSV_COLUMNS = (
    "subject_id",
    "mode",
    "iterations",
    "inpainted_brain_fraction",
    "lesions",
    "lesions_touched",
    "median_contrast_ratio",
    "residual_vesselness",
    "spurious_frst_components",
)


@dataclass(frozen=True)
class IterationRecord:
    """Effect of one dilation setting on one subject."""

    subject_id: str
    mode: str
    iterations: int
    inpainted_brain_fraction: float
    lesions: int
    lesions_touched: int
    median_contrast_ratio: float
    residual_vesselness: float
    spurious_frst_components: int
    contrast_ratios: tuple[float, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class SettingSummary:
    """Effect of one dilation setting pooled over subjects."""

    mode: str
    iterations: int
    inpainted_brain_fraction: float
    touched_fraction: float
    median_contrast_ratio: float
    residual_vesselness: float
    spurious_frst_components: float


@dataclass(frozen=True)
class SubjectContext:
    """Arrays shared by every dilation setting of one subject."""

    volume: np.ndarray
    brain: np.ndarray
    labels: np.ndarray
    centers: list[tuple[int, int, int]]
    contrast_before: list[float]
    spacing: np.ndarray
    vesselness_region: np.ndarray
    near_lesion: np.ndarray


def inspect_dataset(
    dataset_dir: Path,
    bias_field_correction: BiasFieldCorrection,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    frst_threshold: float = DEFAULT_FRST_THRESHOLD,
    subject_count: int | None = None,
    seed: int = 0,
    checkpoint: Path | None = None,
) -> list[IterationRecord]:
    """Measure every dilation setting on the masked subjects of a raw dataset."""
    manifest = RawDatasetManifest.read(
        DatasetLayout(dataset_dir=dataset_dir).raw_manifest_path()
    )
    modalities = {source.source_id: source.modality for source in manifest.sources}
    subjects = [subject for subject in manifest.subjects if subject.mask_path]
    if not subjects:
        raise ApplicationError(
            category="Input data",
            summary="Inpainting inspection requires subjects with masks",
            cause="No subject in raw.json has a mask",
            fix="Index a dataset with microbleed masks",
        )
    if subject_count is not None and subject_count < len(subjects):
        chosen = np.random.default_rng(seed).choice(
            len(subjects), subject_count, replace=False
        )
        subjects = [subjects[index] for index in sorted(chosen)]

    run_settings = {
        "dataset_dir": str(dataset_dir),
        "bias_field_correction": bias_field_correction,
        "max_iterations": max_iterations,
        "frst_threshold": frst_threshold,
        "subjects": [subject.subject_id for subject in subjects],
    }
    finished = report_checkpoints.read(checkpoint, run_settings)
    pending = sum(subject.subject_id not in finished for subject in subjects)
    records: list[IterationRecord] = []
    settings = 1 + len(DILATION_MODES) * max_iterations
    logger.info(
        "Inspecting %d subjects, %d dilation settings each; %d already in checkpoint",
        len(subjects),
        settings,
        len(subjects) - pending,
    )
    started = time.monotonic()
    computed = 0
    for index, subject in enumerate(subjects, start=1):
        if subject.subject_id in finished:
            records.extend(
                record_from_row(row) for row in finished[subject.subject_id]
            )
            logger.info(
                "[%d/%d] %s: loaded from checkpoint",
                index,
                len(subjects),
                subject.subject_id,
            )
            continue
        subject_started = time.monotonic()
        logger.info(
            "[%d/%d] %s: preprocessing up to inpainting",
            index,
            len(subjects),
            subject.subject_id,
        )
        with subject_failure_context(subject.subject_id, "inpainting inspection"):
            prepared = processor.prepare_volume(
                PreprocessInput(
                    io.load_volume(subject.volume_path),
                    io.load_volume(cast(str, subject.mask_path)),
                    modalities[subject.source_id],
                    bias_field_correction,
                )
            )
            subject_records = inspect_subject(
                subject.subject_id, prepared, max_iterations, frst_threshold
            )
        records.extend(subject_records)
        report_checkpoints.append(
            checkpoint,
            run_settings,
            subject.subject_id,
            [asdict(item) for item in subject_records],
        )
        computed += 1
        elapsed = time.monotonic() - started
        remaining = elapsed / computed * (pending - computed)
        logger.info(
            "[%d/%d] %s: done in %s; about %s remaining",
            index,
            len(subjects),
            subject.subject_id,
            format_duration(time.monotonic() - subject_started),
            format_duration(remaining),
        )
    return records


def record_from_row(row: dict) -> IterationRecord:
    """Rebuild a record read back from a checkpoint."""
    return IterationRecord(**{**row, "contrast_ratios": tuple(row["contrast_ratios"])})


def format_duration(seconds: float) -> str:
    """Format a duration as hours, minutes, and seconds."""
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes}m{secs:02d}s"


def inspect_subject(
    subject_id: str,
    prepared: PreprocessOutput,
    max_iterations: int,
    frst_threshold: float,
) -> list[IterationRecord]:
    """Measure inpainting for every dilation mode and iteration count."""
    volume = np.asarray(prepared.volume, dtype=float)
    brain = prepared.brain_mask > 0
    spacing = np.asarray(nib.affines.voxel_sizes(prepared.affine), dtype=float)
    labels = label_components(
        cast(np.ndarray, prepared.mask) > 0, COMPONENT_CONNECTIVITY
    )
    centers = patch.get_target_centers(labels)
    # With no lesions, the distance transform has no zeros to measure from.
    near_lesion = (
        np.asarray(distance_transform_edt(labels == 0, sampling=spacing))
        <= LESION_EXCLUSION_MM
        if centers
        else np.zeros(volume.shape, dtype=bool)
    )
    logger.info("%s: finding vessels", subject_id)
    vessels = inpaint_vessels.get_volume_vessel_mask(volume) > 0
    # The widest setting bounds every other one, so all settings share a region.
    vesselness_region = (
        dilate(vessels, "3d", max_iterations) & brain & ~near_lesion
    )
    context = SubjectContext(
        volume=volume,
        brain=brain,
        labels=labels,
        centers=centers,
        contrast_before=[
            local_contrast(volume, center, spacing, brain, True)
            for center in centers
        ],
        spacing=spacing,
        vesselness_region=vesselness_region,
        near_lesion=near_lesion,
    )

    settings = 1 + len(DILATION_MODES) * max_iterations
    logger.info("%s: setting 1/%d (no dilation)", subject_id, settings)
    undilated = measure(context, vessels, frst_threshold)
    records = []
    done = 1
    for mode in DILATION_MODES:
        records.append(record(subject_id, mode, 0, context, undilated))
        for iterations in range(1, max_iterations + 1):
            done += 1
            logger.info(
                "%s: setting %d/%d (%s, k=%d)",
                subject_id,
                done,
                settings,
                mode,
                iterations,
            )
            measures = measure(
                context, dilate(vessels, mode, iterations), frst_threshold
            )
            records.append(record(subject_id, mode, iterations, context, measures))
    return records


def dilate(mask: np.ndarray, mode: str, iterations: int) -> np.ndarray:
    """Dilate a mask at least once, in 3D or within each slice."""
    # Never pass iterations < 1: scipy then dilates until the mask stops changing.
    structure = generate_binary_structure(3, 1)
    if mode == "in_plane":
        structure[:, :, [0, 2]] = False
    return binary_dilation(mask, structure=structure, iterations=iterations)


def measure(
    context: SubjectContext, inpaint_mask: np.ndarray, frst_threshold: float
) -> tuple[float, int, tuple[float, ...], float, int]:
    """Inpaint one mask and measure tissue cost, lesion damage, and residue."""
    inpainted = inpaint_vessels.inpaint_with_neighborhood_mean(
        context.volume, inpaint_mask
    )
    brain_fraction = float((inpaint_mask & context.brain).sum() / context.brain.sum())
    touched = int(np.unique(context.labels[inpaint_mask & (context.labels > 0)]).size)
    ratios = []
    for center, before in zip(context.centers, context.contrast_before):
        after = local_contrast(inpainted, center, context.spacing, context.brain, True)
        if before > 0 and math.isfinite(after):
            ratios.append(after / before)
    residual = (
        float(vesselness(inpainted)[context.vesselness_region].mean())
        if context.vesselness_region.any()
        else math.nan
    )
    frst_map = frst.apply(inpainted, context.brain)
    spurious = (frst_map >= frst_threshold) & context.brain & ~context.near_lesion
    components = int(label_components(spurious, COMPONENT_CONNECTIVITY).max())
    return brain_fraction, touched, tuple(ratios), residual, components


def record(
    subject_id: str,
    mode: str,
    iterations: int,
    context: SubjectContext,
    measures: tuple[float, int, tuple[float, ...], float, int],
) -> IterationRecord:
    """Pack one setting's measures into a record."""
    brain_fraction, touched, ratios, residual, components = measures
    return IterationRecord(
        subject_id=subject_id,
        mode=mode,
        iterations=iterations,
        inpainted_brain_fraction=brain_fraction,
        lesions=len(context.centers),
        lesions_touched=touched,
        median_contrast_ratio=median_or_nan(ratios),
        residual_vesselness=residual,
        spurious_frst_components=components,
        contrast_ratios=ratios,
    )


def vesselness(volume: np.ndarray) -> np.ndarray:
    """Compute the pipeline's slice-wise Frangi vesselness for a volume."""
    # Sequential on purpose: parallel workers crashed (SIGSEGV) on real data.
    return np.stack(
        [slice_vesselness(volume[:, :, index]) for index in range(volume.shape[2])],
        axis=2,
    )


def slice_vesselness(image_slice: np.ndarray) -> np.ndarray:
    """Compute Frangi vesselness for one slice with the pipeline settings."""
    if np.min(image_slice) == np.max(image_slice):
        return np.zeros_like(image_slice)
    return frangi(
        image_slice,
        sigmas=FRANGI_SIGMAS,  # pyright: ignore[reportArgumentType]
        beta=FRANGI_BETA,
        alpha=FRANGI_ALPHA,
        black_ridges=FRANGI_BLACK_RIDGES,
    )


def summarize(records: Sequence[IterationRecord], mode: str) -> list[SettingSummary]:
    """Pool one mode's records over subjects for each iteration count."""
    summaries = []
    for iterations in sorted({item.iterations for item in records}):
        group = [
            item
            for item in records
            if item.mode == mode and item.iterations == iterations
        ]
        lesions = sum(item.lesions for item in group)
        summaries.append(
            SettingSummary(
                mode=mode,
                iterations=iterations,
                inpainted_brain_fraction=mean_or_nan(
                    [item.inpainted_brain_fraction for item in group]
                ),
                touched_fraction=(
                    sum(item.lesions_touched for item in group) / lesions
                    if lesions
                    else 0.0
                ),
                median_contrast_ratio=median_or_nan(
                    [ratio for item in group for ratio in item.contrast_ratios]
                ),
                residual_vesselness=mean_or_nan(
                    [item.residual_vesselness for item in group]
                ),
                spurious_frst_components=mean_or_nan(
                    [float(item.spurious_frst_components) for item in group]
                ),
            )
        )
    return summaries


def recommend(summaries: Sequence[SettingSummary]) -> tuple[int, str]:
    """Pick the smallest iteration count that removes most vessel signal."""
    baseline = summaries[0]
    allowed = [
        item
        for item in summaries
        if item.touched_fraction - baseline.touched_fraction
        <= MAXIMUM_EXTRA_TOUCHED
    ]
    finite = [
        item.residual_vesselness
        for item in summaries
        if math.isfinite(item.residual_vesselness)
    ]
    if not math.isfinite(baseline.residual_vesselness) or (
        baseline.residual_vesselness - min(finite) <= 0
    ):
        return 0, "dilation does not reduce residual vesselness"
    possible = baseline.residual_vesselness - min(finite)

    def reduction(item: SettingSummary) -> float:
        return (baseline.residual_vesselness - item.residual_vesselness) / possible

    for item in allowed:
        if reduction(item) >= REDUCTION_TARGET:
            return item.iterations, (
                f"smallest count reaching {REDUCTION_TARGET:.0%} of the maximum "
                "vesselness reduction"
            )
    best = max(allowed, key=reduction)
    return best.iterations, (
        f"largest reduction ({reduction(best):.0%}) while touching at most "
        f"{MAXIMUM_EXTRA_TOUCHED:.0%} more lesions than no dilation"
    )


def write_csv(records: Sequence[IterationRecord], output: TextIO) -> None:
    """Write one CSV row per subject, mode, and iteration count."""
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for item in records:
        values = (getattr(item, column) for column in CSV_COLUMNS)
        writer.writerow(
            round(value, 4) if isinstance(value, float) else value
            for value in values
        )


def write_summary(records: Sequence[IterationRecord], output: TextIO) -> None:
    """Write pooled tables and the recommended setting for each mode."""
    choices = []
    for mode in DILATION_MODES:
        summaries = summarize(records, mode)
        iterations, reason = recommend(summaries)
        chosen = summaries[iterations]
        choices.append(chosen)
        output.write(f"\nmode={mode}\n")
        output.write(
            f"{'k':>3}{'brain_inpainted':>17}{'lesions_touched':>17}"
            f"{'contrast_kept':>15}{'vesselness':>12}{'frst_spurious':>15}\n"
        )
        for item in summaries:
            output.write(
                f"{item.iterations:>3}{item.inpainted_brain_fraction:>17.4f}"
                f"{item.touched_fraction:>17.3f}{item.median_contrast_ratio:>15.3f}"
                f"{item.residual_vesselness:>12.5f}"
                f"{item.spurious_frst_components:>15.1f}\n"
            )
        output.write(f"recommended k={iterations}: {reason}\n")
    best = min(
        choices,
        key=lambda item: (item.touched_fraction, item.inpainted_brain_fraction),
    )
    output.write(f"\noverall: mode={best.mode} iterations={best.iterations}\n")


def median_or_nan(values: Sequence[float]) -> float:
    """Return the median of the finite values, or NaN when there are none."""
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else math.nan


def mean_or_nan(values: Sequence[float]) -> float:
    """Return the mean of the finite values, or NaN when there are none."""
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def main(arguments: Sequence[str] | None = None) -> None:
    """Run the vessel-dilation inspection."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure how vessel-mask dilation before inpainting trades vessel "
            "removal against lesion damage, and recommend an iteration count."
        )
    )
    parser.add_argument(
        "dataset_dir", type=Path, help="Dataset directory with manifests/raw.json"
    )
    parser.add_argument(
        "--bias-field-correction",
        choices=("fast", "n4"),
        required=True,
        help="Same method as configs/preprocess.toml",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS
    )
    parser.add_argument(
        "--subjects", type=int, help="Inspect a random subset of masked subjects"
    )
    parser.add_argument("--seed", type=int, default=0, help="Subset seed")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="JSON-lines file that saves each finished subject; rerun to resume",
    )
    parser.add_argument(
        "--frst-threshold",
        type=float,
        default=DEFAULT_FRST_THRESHOLD,
        help="FRST level counted as a spurious blob",
    )
    options = parser.parse_args(arguments)
    if options.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    records = inspect_dataset(
        options.dataset_dir,
        options.bias_field_correction,
        options.max_iterations,
        options.frst_threshold,
        options.subjects,
        options.seed,
        options.checkpoint,
    )
    write_csv(records, sys.stdout)
    write_summary(records, sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()
