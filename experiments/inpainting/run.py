"""Compare vessel detection strategies and in-plane dilation iteration counts.

uv run python -m experiments.inpainting.run --config experiments/inpainting/config.toml
"""

import argparse
import csv
import random
import time
import traceback
from collections.abc import Sequence
from pathlib import Path

from microbleednet.core.datamodels import Modality
from microbleednet.orchestration.layouts import DatasetLayout
from microbleednet.orchestration.manifests import RawDatasetManifest, RawSubject

from . import export
from .config import InpaintingConfig, load_config
from .log import Log, RunLog, silent
from .measure import Measurement, measure
from .prepare import prepare_subject
from .summary import recommend, summarize_iterations

Rows = dict[str, list[dict[str, float]]]


def select_subjects(
    subjects: Sequence[RawSubject], max_subjects: int | None, seed: int
) -> list[RawSubject]:
    """Keep subjects with a lesion mask, then take a seeded random subset."""
    masked = sorted(
        (subject for subject in subjects if subject.mask_path),
        key=lambda subject: subject.subject_id,
    )
    if max_subjects is None or max_subjects >= len(masked):
        return masked
    chosen = set(random.Random(seed).sample(range(len(masked)), max_subjects))
    return [subject for index, subject in enumerate(masked) if index in chosen]


def evaluate(
    config: InpaintingConfig,
    manifest: RawDatasetManifest,
    subjects: Sequence[RawSubject],
    log: Log = silent,
) -> Rows:
    """Measure every strategy and iteration count on every subject.

    Returns one row per iteration count for each strategy. Progress is reported
    to ``log``.
    """
    modalities: dict[str, Modality] = {
        source.source_id: source.modality for source in manifest.sources
    }
    cells = [(s, n) for s in config.strategies for n in config.iterations]
    measurements: dict[tuple[str, int], list[Measurement]] = {c: [] for c in cells}
    seconds = dict.fromkeys(cells, 0.0)
    cache_dir = config.output_dir / "cache"
    run_started = time.perf_counter()

    for position, subject in enumerate(subjects, start=1):
        subject_started = time.perf_counter()
        log(f"[{position}/{len(subjects)}] {subject.subject_id}")
        prepared = prepare_subject(
            subject,
            modalities[subject.source_id],
            config.bias_field_correction,
            config.strategies,
            cache_dir,
            log,
        )
        directory = export.subject_directory(config.output_dir, subject.subject_id)
        if config.save_masks:
            export.save_subject_inputs(prepared, directory)
        for strategy, iterations in cells:
            started = time.perf_counter()
            outcome = measure(
                prepared,
                strategy,
                iterations,
                lesion_margin=config.lesion_margin,
                exclusion_radius=config.exclusion_radius,
                peak_min_distance=config.peak_min_distance,
            )
            elapsed = time.perf_counter() - started
            measurements[strategy, iterations].append(outcome.measurement)
            seconds[strategy, iterations] += elapsed
            log(
                f"  {strategy} n={iterations}: "
                f"{describe(outcome.measurement)}, {elapsed:.1f}s"
            )
            if config.save_masks:
                export.save_outcome(
                    prepared,
                    directory,
                    strategy,
                    iterations,
                    outcome.mask,
                    outcome.inpainted if config.save_inpainted else None,
                )
        if config.save_masks:
            log(f"  masks saved to {directory}")
        done = time.perf_counter() - run_started
        remaining = done / position * (len(subjects) - position)
        log(
            f"  subject done in {time.perf_counter() - subject_started:.0f}s; "
            f"elapsed {format_duration(done)}, about {format_duration(remaining)} left"
        )

    return build_rows(config, cells, measurements, seconds, len(subjects))


def build_rows(
    config: InpaintingConfig,
    cells: Sequence[tuple[str, int]],
    measurements: dict[tuple[str, int], list[Measurement]],
    seconds: dict[tuple[str, int], float],
    subject_count: int,
) -> Rows:
    """Pool per-subject measurements into one row per strategy and iteration count."""
    rows: Rows = {strategy: [] for strategy in config.strategies}
    for strategy, iterations in cells:
        row = {"iterations": float(iterations)}
        row.update(
            summarize_iterations(measurements[strategy, iterations], config.recall_at)
        )
        row["subjects"] = float(subject_count)
        row["seconds"] = round(seconds[strategy, iterations], 2)
        rows[strategy].append(row)
    return rows


def describe(measurement: Measurement) -> str:
    """Summarize one subject's measurement for the progress log."""
    return (
        f"lesion voxels erased {measurement.lesion_voxels_removed}/"
        f"{measurement.lesion_voxels}, lesions hit {measurement.lesions_hit}/"
        f"{measurement.lesions}, {measurement.vessel_fraction:.2%} of brain inpainted"
    )


def format_duration(seconds: float) -> str:
    """Format a duration as hours, minutes and seconds."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def write_results(
    output_dir: Path,
    rows: Rows,
    recommended: dict[str, int | None],
    name: str = "results.csv",
) -> None:
    """Write one table for all strategies, marking each one's recommended count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    first = next(iter(rows.values()))[0]
    columns = ["strategy", *first, "recommended"]
    with (output_dir / name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for strategy, strategy_rows in rows.items():
            for row in strategy_rows:
                writer.writerow(
                    {
                        "strategy": strategy,
                        **row,
                        "recommended": int(row["iterations"]) == recommended[strategy],
                    }
                )


def format_table(rows: Rows, recall_at: Sequence[int]) -> str:
    """Render one table per strategy as plain text."""
    metrics = [
        "lesion_loss",
        "lesions_hit",
        "vessel_fraction",
        "residual_vessels",
        "mean_auc",
        *(f"recall@{count}" for count in recall_at),
    ]
    header = f"{'iterations':>10}  " + "  ".join(f"{m:>19}" for m in metrics)
    blocks = []
    for strategy, strategy_rows in rows.items():
        lines = [f"Strategy: {strategy}", header]
        for row in strategy_rows:
            values = "  ".join(f"{row[metric]:>19.3f}" for metric in metrics)
            lines.append(f"{int(row['iterations']):>10}  {values}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment, or only show what it would use with ``--dry-run``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="List the subjects and exit."
    )
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    manifest = RawDatasetManifest.read(
        DatasetLayout(dataset_dir=config.dataset_dir).raw_manifest_path()
    )
    subjects = select_subjects(manifest.subjects, config.max_subjects, config.seed)
    if not subjects:
        print("No subjects with lesion masks in the dataset.")
        return 1
    if arguments.dry_run:
        print(f"strategies: {list(config.strategies)}")
        print(f"iterations: {list(config.iterations)}")
        print(f"subjects ({len(subjects)}): {[s.subject_id for s in subjects]}")
        return 0

    log = RunLog(config.output_dir / "run.log")
    log(
        f"dataset: {config.dataset_dir} "
        f"(bias field correction: {config.bias_field_correction})"
    )
    log(f"strategies: {list(config.strategies)}; iterations: {list(config.iterations)}")
    log(f"subjects: {len(subjects)}; masks saved: {config.save_masks}")
    try:
        rows = evaluate(config, manifest, subjects, log)
    except Exception:
        log("FAILED; the run stopped with this error:")
        log.raw(traceback.format_exc())
        raise
    recommended = {
        strategy: recommend(strategy_rows, config.max_lesion_loss, config.auc_tolerance)
        for strategy, strategy_rows in rows.items()
    }
    write_results(config.output_dir, rows, recommended)
    log.raw("")
    log.raw(format_table(rows, config.recall_at))
    log.raw("")
    for strategy, count in recommended.items():
        if count is None:
            log(
                f"{strategy}: no iteration count keeps lesion loss within "
                f"{config.max_lesion_loss:.0%}; widen max_lesion_loss or inspect "
                "the data."
            )
        else:
            log(f"{strategy}: recommended in-plane dilation iterations: {count}")
    log(f"wrote {config.output_dir / 'results.csv'}")
    if config.save_masks:
        log(f"wrote masks to {config.output_dir / 'masks'}")
    log(f"log: {log.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
