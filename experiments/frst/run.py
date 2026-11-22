"""Rank FRST variants on a preprocessed dataset.

uv run python -m experiments.frst.run --config experiments/frst/config.toml
"""

import argparse
import csv
import json
import random
import time
import traceback
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from microbleednet.core import io
from microbleednet.orchestration.layouts import DatasetLayout
from microbleednet.orchestration.manifests import (
    PreprocessedDatasetManifest,
    PreprocessedSubject,
)

from . import scoring, variants
from .config import ExperimentConfig, expand_variants, load_config
from .log import Log, RunLog, format_duration, silent


def select_subjects(
    subjects: Sequence[PreprocessedSubject], max_subjects: int | None, seed: int
) -> list[PreprocessedSubject]:
    """Keep subjects with a lesion mask, then take a seeded random subset."""
    masked = sorted(
        (subject for subject in subjects if subject.variants[0].mask_path),
        key=lambda subject: subject.subject_id,
    )
    if max_subjects is None or max_subjects >= len(masked):
        return masked
    chosen = set(random.Random(seed).sample(range(len(masked)), max_subjects))
    return [subject for index, subject in enumerate(masked) if index in chosen]


def load_subject(
    subject: PreprocessedSubject,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Load the volume, lesion mask and brain mask of the unaugmented variant."""
    variant = subject.variants[0]
    assert variant.mask_path is not None
    image = io.load_volume(variant.volume_path)
    x, y, z = (float(size) for size in image.header.get_zooms()[:3])
    spacing = (x, y, z)
    return (
        io.nifti_to_numpy(image),
        io.nifti_to_numpy(io.load_volume(variant.mask_path)),
        io.nifti_to_numpy(io.load_volume(subject.brain_mask_path)),
        spacing,
    )


def evaluate(
    config: ExperimentConfig,
    named_specs: Sequence[tuple[str, variants.FrstSpec]],
    subjects: Sequence[PreprocessedSubject],
    log: Log = silent,
) -> list[dict[str, object]]:
    """Score every variant on every subject and return one row per variant.

    Progress is reported to ``log``: every subject, every variant on it and an
    estimate of the time left.
    """
    per_variant: dict[str, list[scoring.SubjectScores]] = {
        name: [] for name, _ in named_specs
    }
    seconds = dict.fromkeys(per_variant, 0.0)
    run_started = time.perf_counter()
    for position, subject in enumerate(subjects, start=1):
        subject_started = time.perf_counter()
        log(f"[{position}/{len(subjects)}] {subject.subject_id}")
        volume, lesions, brain, spacing = load_subject(subject)
        log(
            f"  loaded in {time.perf_counter() - subject_started:.1f}s: "
            f"shape {volume.shape}, spacing {tuple(round(s, 2) for s in spacing)}"
        )
        for index, (name, spec) in enumerate(named_specs, start=1):
            started = time.perf_counter()
            frst_map = variants.apply(volume, brain, spacing, spec)
            scores = scoring.subject_scores(
                frst_map,
                lesions,
                brain,
                lesion_margin=config.lesion_margin,
                exclusion_radius=config.exclusion_radius,
                peak_min_distance=config.peak_min_distance,
            )
            per_variant[name].append(scores)
            elapsed = time.perf_counter() - started
            seconds[name] += elapsed
            log(
                f"  [{index}/{len(named_specs)}] {name}: "
                f"{len(scores.lesion_scores)} lesions, "
                f"{len(scores.background_peaks)} background peaks, {elapsed:.1f}s"
            )
        done = time.perf_counter() - run_started
        remaining = done / position * (len(subjects) - position)
        log(
            f"  subject done in {time.perf_counter() - subject_started:.0f}s; "
            f"elapsed {format_duration(done)}, about {format_duration(remaining)} left"
        )

    rows: list[dict[str, object]] = []
    for name, spec in named_specs:
        row: dict[str, object] = {"name": name}
        row.update(scoring.summarize(per_variant[name], config.recall_at))
        row["subjects"] = len(subjects)
        row["seconds"] = round(seconds[name], 2)
        row.update(
            {
                field: json.dumps(value) if isinstance(value, tuple) else value
                for field, value in spec.model_dump().items()
            }
        )
        rows.append(row)
    return sorted(rows, key=_ranking_key, reverse=True)


def write_results(
    output_dir: Path,
    rows: Sequence[dict[str, object]],
    named_specs: Sequence[tuple[str, variants.FrstSpec]],
) -> None:
    """Write the ranked table and the exact variant specifications."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    specs = {name: spec.model_dump(mode="json") for name, spec in named_specs}
    (output_dir / "variants.json").write_text(json.dumps(specs, indent=2) + "\n")


def format_table(rows: Sequence[dict[str, object]], recall_at: Sequence[int]) -> str:
    """Render the ranking as plain text."""
    metrics = ["mean_auc", *(f"recall@{count}" for count in recall_at), "seconds"]
    width = max(len(str(row["name"])) for row in rows)
    lines = [f"{'variant':<{width}}  " + "  ".join(f"{m:>10}" for m in metrics)]
    for row in rows:
        values = "  ".join(f"{_number(row[metric]):>10}" for metric in metrics)
        lines.append(f"{row['name']!s:<{width}}  {values}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment, or only list its variants with ``--dry-run``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="List the variants and exit."
    )
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    named_specs = expand_variants(config)
    if arguments.dry_run:
        for name, spec in named_specs:
            print(f"{name}: {spec.model_dump(mode='json')}")
        return 0

    manifest = PreprocessedDatasetManifest.read(
        DatasetLayout(dataset_dir=config.dataset_dir).preprocessed_manifest_path()
    )
    subjects = select_subjects(manifest.subjects, config.max_subjects, config.seed)
    if not subjects:
        print("No subjects with lesion masks in the dataset.")
        return 1

    log = RunLog(config.output_dir / "run.log")
    log(f"dataset: {config.dataset_dir}")
    log(f"variants: {len(named_specs)}; subjects: {len(subjects)}")
    try:
        rows = evaluate(config, named_specs, subjects, log)
    except Exception:
        log("FAILED; the run stopped with this error:")
        log.raw(traceback.format_exc())
        raise
    write_results(config.output_dir, rows, named_specs)
    log.raw("")
    log.raw(format_table(rows, config.recall_at))
    log.raw("")
    log(f"wrote {config.output_dir / 'results.csv'}")
    log(f"log: {log.path}")
    return 0


def _ranking_key(row: dict[str, object]) -> tuple[float, ...]:
    """Sort by mean AUC, then by each recall@K; NaN (no lesions) sorts last."""
    keys = ["mean_auc", *(key for key in row if str(key).startswith("recall@"))]
    return tuple(
        float(value) if isinstance(value, float) and value == value else -1.0
        for value in (row[key] for key in keys)
    )


def _number(value: object) -> str:
    """Format a metric for the console table."""
    return f"{value:.3f}" if isinstance(value, float) else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
