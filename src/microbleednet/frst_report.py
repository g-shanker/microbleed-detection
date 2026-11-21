"""Score FRST variants by how well FRST alone separates microbleeds.

Experimental diagnostic. See EXPERIMENTAL_CHANGES.md for how to remove it.
"""

import argparse
import csv
import logging
import math
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence, TextIO

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter

from . import report_checkpoints
from .constants import COMPONENT_CONNECTIVITY, VARIANT_INDEX
from .core import io
from .core.transforms import frst_variants, patch
from .core.transforms.frst_variants import FrstSettings
from .core.utils import label_components
from .errors import ApplicationError
from .inpainting_report import format_duration
from .lesion_report import local_distances, max_within_radius
from .orchestration.layouts import DatasetLayout
from .orchestration.manifests import PreprocessedDatasetManifest
from .orchestration.utils import subject_failure_context

logger = logging.getLogger(__name__)

LESION_RADIUS_MM = 2.0
PEAK_RADIUS_MM = 2.0
BACKGROUND_EXCLUSION_MM = 5.0
FALSE_POSITIVE_RATES = (1, 5, 10, 20)
MINIMUM_FROC_GAIN = 0.02
CSV_COLUMNS = (
    "subject_id",
    "variant",
    "spacing_mm",
    "lesions",
    "background_peaks",
    "median_lesion_score",
    "median_background_peak",
    "subject_auc",
)


@dataclass(frozen=True)
class SubjectScores:
    """FRST scores of one subject's lesions and background peaks."""

    subject_id: str
    variant: str
    spacing: tuple[float, float, float]
    lesion_scores: np.ndarray
    background_peaks: np.ndarray


@dataclass(frozen=True)
class VariantSummary:
    """Pooled detection quality of FRST alone for one variant."""

    variant: str
    sensitivities: tuple[float, ...]
    froc: float
    auc: float


def resolve_variants(
    names: Sequence[str] | None, combinations: Sequence[str]
) -> dict[str, FrstSettings]:
    """Return the baseline, chosen single variants, and requested combinations."""
    chosen = list(frst_variants.VARIANT_OVERRIDES) if names is None else list(names)
    settings = {"baseline": frst_variants.BASELINE}
    for name in chosen:
        settings[name] = frst_variants.settings_for([name])
    for combination in combinations:
        settings[combination] = frst_variants.settings_for(combination.split("+"))
    return settings


def inspect_dataset(
    dataset_dir: Path,
    variants: dict[str, FrstSettings],
    subject_count: int | None = None,
    seed: int = 0,
    device: str = "cpu",
    checkpoint: Path | None = None,
) -> list[SubjectScores]:
    """Score every variant on the masked subjects of a preprocessed dataset."""
    manifest = PreprocessedDatasetManifest.read(
        DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path()
    )
    subjects = [
        subject
        for subject in manifest.subjects
        if subject.variants[VARIANT_INDEX].mask_path
    ]
    if not subjects:
        raise ApplicationError(
            category="Input data",
            summary="FRST comparison requires subjects with masks",
            cause="No subject in preprocessed.json has a variant-0 mask",
            fix="Preprocess a dataset with microbleed masks",
        )
    if subject_count is not None and subject_count < len(subjects):
        chosen = np.random.default_rng(seed).choice(
            len(subjects), subject_count, replace=False
        )
        subjects = [subjects[index] for index in sorted(chosen)]

    run_settings = {
        "dataset_dir": str(dataset_dir),
        "variants": {name: asdict(settings) for name, settings in variants.items()},
        "subjects": [subject.subject_id for subject in subjects],
    }
    finished = report_checkpoints.read(checkpoint, run_settings)
    pending = sum(subject.subject_id not in finished for subject in subjects)
    logger.info(
        "Scoring %d FRST variants on %d subjects; %d already in checkpoint",
        len(variants),
        len(subjects),
        len(subjects) - pending,
    )
    scores: list[SubjectScores] = []
    started = time.monotonic()
    computed = 0
    for index, subject in enumerate(subjects, start=1):
        if subject.subject_id in finished:
            scores.extend(scores_from_row(row) for row in finished[subject.subject_id])
            logger.info(
                "[%d/%d] %s: loaded from checkpoint",
                index,
                len(subjects),
                subject.subject_id,
            )
            continue
        subject_started = time.monotonic()
        subject_scores: list[SubjectScores] = []
        with subject_failure_context(subject.subject_id, "FRST comparison"):
            variant = subject.variants[VARIANT_INDEX]
            image = io.load_volume(variant.volume_path)
            volume = io.nifti_to_numpy(image)
            mask = io.nifti_to_numpy(io.load_volume(str(variant.mask_path)))
            brain = io.nifti_to_numpy(io.load_volume(subject.brain_mask_path)) > 0
            sizes = nib.affines.voxel_sizes(image.affine)
            spacing = (float(sizes[0]), float(sizes[1]), float(sizes[2]))
            logger.info(
                "[%d/%d] %s: spacing %s mm",
                index,
                len(subjects),
                subject.subject_id,
                spacing_label(spacing),
            )
            labels = label_components(mask > 0, COMPONENT_CONNECTIVITY)
            centers = patch.get_target_centers(labels)
            far_from_lesions = (
                np.asarray(distance_transform_edt(labels == 0, sampling=spacing))
                > BACKGROUND_EXCLUSION_MM
                if centers
                else np.ones(volume.shape, dtype=bool)
            )
            for number, (name, settings) in enumerate(variants.items(), start=1):
                logger.info(
                    "%s: variant %d/%d (%s)",
                    subject.subject_id,
                    number,
                    len(variants),
                    name,
                )
                frst_map = frst_variants.compute(
                    volume, brain, spacing, settings, device
                )
                lesion_scores, background = score_map(
                    frst_map, centers, brain & far_from_lesions, spacing
                )
                subject_scores.append(
                    SubjectScores(
                        subject.subject_id, name, spacing, lesion_scores, background
                    )
                )
        scores.extend(subject_scores)
        report_checkpoints.append(
            checkpoint,
            run_settings,
            subject.subject_id,
            [row_from_scores(item) for item in subject_scores],
        )
        computed += 1
        elapsed = time.monotonic() - started
        logger.info(
            "[%d/%d] %s: done in %s; about %s remaining",
            index,
            len(subjects),
            subject.subject_id,
            format_duration(time.monotonic() - subject_started),
            format_duration(elapsed / computed * (pending - computed)),
        )
    return scores


def row_from_scores(item: SubjectScores) -> dict[str, Any]:
    """Convert scores to JSON-friendly values for a checkpoint."""
    return {
        "subject_id": item.subject_id,
        "variant": item.variant,
        "spacing": list(item.spacing),
        "lesion_scores": item.lesion_scores.tolist(),
        "background_peaks": item.background_peaks.tolist(),
    }


def scores_from_row(row: dict[str, Any]) -> SubjectScores:
    """Rebuild scores read back from a checkpoint."""
    x, y, z = row["spacing"]
    return SubjectScores(
        row["subject_id"],
        row["variant"],
        (float(x), float(y), float(z)),
        np.asarray(row["lesion_scores"], dtype=float),
        np.asarray(row["background_peaks"], dtype=float),
    )


def score_map(
    frst_map: np.ndarray,
    centers: list[tuple[int, int, int]],
    background_region: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return each lesion's best nearby FRST and the background peak values."""
    spacing_mm = np.asarray(spacing, dtype=float)
    lesion_scores = np.array(
        [
            max_within_radius(frst_map, center, spacing_mm, LESION_RADIUS_MM)[0]
            for center in centers
        ],
        dtype=float,
    )
    peaks = frst_map == maximum_filter(
        frst_map, footprint=peak_footprint(spacing_mm), mode="nearest"
    )
    # Flat regions are maxima everywhere; only values above the floor count.
    floor = frst_map[background_region].min() if background_region.any() else 0.0
    peaks &= background_region & (frst_map > floor)
    return lesion_scores, frst_map[peaks].astype(float)


def peak_footprint(spacing: np.ndarray) -> np.ndarray:
    """Return an ellipsoid of PEAK_RADIUS_MM in voxels for local maxima."""
    half = np.ceil(PEAK_RADIUS_MM / spacing).astype(int)
    shape = tuple(int(2 * value + 1) for value in half)
    center = (int(half[0]), int(half[1]), int(half[2]))
    window, distance = local_distances(shape, center, spacing, PEAK_RADIUS_MM)
    footprint = np.zeros(shape, dtype=bool)
    footprint[window] = distance <= PEAK_RADIUS_MM
    return footprint


def froc(
    lesion_scores: np.ndarray, background: np.ndarray, subject_count: int
) -> tuple[float, ...]:
    """Return lesion sensitivity at each false-positive rate per subject."""
    if lesion_scores.size == 0:
        return tuple(math.nan for _ in FALSE_POSITIVE_RATES)
    ordered = np.sort(background)[::-1]
    sensitivities = []
    for rate in FALSE_POSITIVE_RATES:
        allowed = rate * subject_count
        # A lesion counts when it beats the first background peak not allowed.
        threshold = ordered[allowed] if allowed < ordered.size else -math.inf
        sensitivities.append(float(np.mean(lesion_scores > threshold)))
    return tuple(sensitivities)


def auc(lesion_scores: np.ndarray, background: np.ndarray) -> float:
    """Return the chance a lesion outscores a random background peak."""
    if lesion_scores.size == 0 or background.size == 0:
        return math.nan
    ordered = np.sort(background)
    below = np.searchsorted(ordered, lesion_scores, side="left")
    not_above = np.searchsorted(ordered, lesion_scores, side="right")
    return float(np.mean((below + not_above) / 2) / ordered.size)


def summarize(scores: Sequence[SubjectScores]) -> list[VariantSummary]:
    """Pool scores over subjects for every variant, best first."""
    summaries = []
    for variant in dict.fromkeys(item.variant for item in scores):
        group = [item for item in scores if item.variant == variant]
        lesions = np.concatenate([item.lesion_scores for item in group])
        background = np.concatenate([item.background_peaks for item in group])
        sensitivities = froc(lesions, background, len(group))
        summaries.append(
            VariantSummary(
                variant=variant,
                sensitivities=sensitivities,
                froc=float(np.mean(sensitivities)),
                auc=auc(lesions, background),
            )
        )
    return sorted(
        summaries,
        key=lambda item: -item.froc if math.isfinite(item.froc) else math.inf,
    )


def spacing_label(spacing: tuple[float, float, float]) -> str:
    """Format a voxel spacing such as 0.50x0.50x2.00."""
    return "x".join(f"{value:.2f}" for value in spacing)


def write_csv(scores: Sequence[SubjectScores], output: TextIO) -> None:
    """Write one CSV row per subject and variant."""
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for item in scores:
        writer.writerow(
            (
                item.subject_id,
                item.variant,
                spacing_label(item.spacing),
                item.lesion_scores.size,
                item.background_peaks.size,
                round(median(item.lesion_scores), 4),
                round(median(item.background_peaks), 4),
                round(auc(item.lesion_scores, item.background_peaks), 4),
            )
        )


def write_summary(scores: Sequence[SubjectScores], output: TextIO) -> None:
    """Write the voxel spacings, the pooled variant table, and next steps."""
    spacings = Counter(
        spacing_label(item.spacing)
        for item in {item.subject_id: item for item in scores}.values()
    )
    output.write("\nvoxel spacing (mm): ")
    output.write(", ".join(f"{label} ({count})" for label, count in spacings.items()))
    output.write("\n\n")
    summaries = summarize(scores)
    baseline = next(item for item in summaries if item.variant == "baseline")
    rates = "".join(f"{f'sens@{rate}':>9}" for rate in FALSE_POSITIVE_RATES)
    output.write(f"{'variant':<34}{rates}{'froc':>8}{'auc':>8}{'vs_base':>9}\n")
    for item in summaries:
        values = "".join(f"{value:>9.3f}" for value in item.sensitivities)
        output.write(
            f"{item.variant:<34}{values}{item.froc:>8.3f}{item.auc:>8.3f}"
            f"{item.froc - baseline.froc:>+9.3f}\n"
        )
    better = [
        item.variant
        for item in summaries
        if "+" not in item.variant
        and item.froc - baseline.froc >= MINIMUM_FROC_GAIN
    ]
    if better:
        output.write(
            f"\nvariants beating baseline by >= {MINIMUM_FROC_GAIN}: "
            f"{', '.join(better)}\n"
            f"next: rerun with --combine {'+'.join(better)} to test them together\n"
        )
    else:
        output.write(
            f"\nno single variant beats baseline by >= {MINIMUM_FROC_GAIN}\n"
        )


def median(values: np.ndarray) -> float:
    """Return the median, or NaN for an empty array."""
    return float(np.median(values)) if values.size else math.nan


def main(arguments: Sequence[str] | None = None) -> None:
    """Run the FRST variant comparison."""
    parser = argparse.ArgumentParser(
        description=(
            "Score FRST variants by how well FRST alone separates annotated "
            "microbleeds from background peaks."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Dataset directory with manifests/preprocessed.json",
    )
    parser.add_argument(
        "--variants",
        help=(
            "Comma-separated variants to score (default: all): "
            + ", ".join(frst_variants.VARIANT_OVERRIDES)
        ),
    )
    parser.add_argument(
        "--combine",
        action="append",
        default=[],
        help="Variants joined with '+', scored together; repeatable",
    )
    parser.add_argument(
        "--subjects", type=int, help="Score a random subset of masked subjects"
    )
    parser.add_argument("--seed", type=int, default=0, help="Subset seed")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cuda")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="JSON-lines file that saves each finished subject; rerun to resume",
    )
    options = parser.parse_args(arguments)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    variants = resolve_variants(
        options.variants.split(",") if options.variants else None, options.combine
    )
    scores = inspect_dataset(
        options.dataset_dir,
        variants,
        options.subjects,
        options.seed,
        options.device,
        options.checkpoint,
    )
    write_csv(scores, sys.stdout)
    write_summary(scores, sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()
