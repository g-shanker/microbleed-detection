"""Rank FRST variants on a preprocessed dataset.

uv run python -m experiments.frst.run --config experiments/frst/config.toml
"""

import argparse
import csv
import json
import random
import time
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
    log=print,
) -> list[dict[str, object]]:
    """Score every variant on every subject and return one row per variant."""
    per_variant: dict[str, list[scoring.SubjectScores]] = {
        name: [] for name, _ in named_specs
    }
    seconds = dict.fromkeys(per_variant, 0.0)
    for position, subject in enumerate(subjects, start=1):
        log(f"[{position}/{len(subjects)}] {subject.subject_id}")
        volume, lesions, brain, spacing = load_subject(subject)
        for name, spec in named_specs:
            started = time.perf_counter()
            frst_map = variants.apply(volume, brain, spacing, spec)
            per_variant[name].append(
                scoring.subject_scores(
                    frst_map,
                    lesions,
                    brain,
                    lesion_margin=config.lesion_margin,
                    exclusion_radius=config.exclusion_radius,
                    peak_min_distance=config.peak_min_distance,
                )
            )
            seconds[name] += time.perf_counter() - started

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

    rows = evaluate(config, named_specs, subjects)
    write_results(config.output_dir, rows, named_specs)
    print(format_table(rows, config.recall_at))
    print(f"\nWrote {config.output_dir / 'results.csv'}")
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
