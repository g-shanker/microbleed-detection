"""Look at the subjects a run has finished so far, without waiting for the rest.

uv run python -m experiments.inpainting.analyze \\
    --config experiments/inpainting/config.toml

The run caches every subject's volume, lesions and vessel masks as soon as it
has prepared them. This command scores only those cached subjects with the same
metrics as the full run and prints the same tables, so it can be run while the
full run is still going or after it was stopped. It never preprocesses or
detects anything, so it needs no FSL.

By default it skips ``residual_vessels``, which reruns each detector on every
inpainted volume and is the slowest part of a measurement. The lesion, FRST
separation and inpainted-area metrics do not depend on it. Add ``--residual`` to
include it.
"""

import argparse
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from .config import InpaintingConfig, load_config
from .measure import Measurement, measure
from .prepare import PreparedSubject, load_prepared
from .run import build_rows, format_duration, format_table, write_results
from .summary import recommend

# Below this many lesions the metrics mostly reflect individual cases.
FEW_LESIONS = 10
PARTIAL_RESULTS = "partial_results.csv"
PARTIAL_SUFFIX = ".partial.npz"


def say(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def cached_subjects(
    config: InpaintingConfig, limit: int | None
) -> tuple[list[tuple[str, PreparedSubject]], list[str]]:
    """Load the finished cache entries; also return those that could not be used.

    Entries are in subject order. A subject is unusable when it lacks one of the
    configured strategies' vessel masks or its file cannot be read, for example
    because the run is writing it right now.
    """
    directory = config.output_dir / "cache" / config.bias_field_correction
    paths = sorted(
        path
        for path in directory.glob("*.npz")
        if not path.name.endswith(PARTIAL_SUFFIX)
    )
    usable: list[tuple[str, PreparedSubject]] = []
    skipped: list[str] = []
    for path in paths:
        if limit is not None and len(usable) >= limit:
            break
        prepared = load_prepared(path, config.strategies)
        if prepared is None:
            skipped.append(path.stem)
        else:
            usable.append((path.stem, prepared))
    return usable, skipped


def analyze(
    config: InpaintingConfig,
    subjects: Sequence[tuple[str, PreparedSubject]],
    residual: bool,
    log=say,
):
    """Measure every strategy and iteration count on the given subjects."""
    cells = [(s, n) for s in config.strategies for n in config.iterations]
    measurements: dict[tuple[str, int], list[Measurement]] = {c: [] for c in cells}
    seconds = dict.fromkeys(cells, 0.0)
    started = time.perf_counter()

    for position, (name, prepared) in enumerate(subjects, start=1):
        log(f"[{position}/{len(subjects)}] {name}")
        for strategy, iterations in cells:
            cell_started = time.perf_counter()
            outcome = measure(
                prepared,
                strategy,
                iterations,
                lesion_margin=config.lesion_margin,
                exclusion_radius=config.exclusion_radius,
                peak_min_distance=config.peak_min_distance,
                residual=residual,
            )
            measurements[strategy, iterations].append(outcome.measurement)
            seconds[strategy, iterations] += time.perf_counter() - cell_started
        elapsed = time.perf_counter() - started
        left = elapsed / position * (len(subjects) - position)
        log(
            f"  done; elapsed {format_duration(elapsed)}, "
            f"about {format_duration(left)} left"
        )
    return build_rows(config, cells, measurements, seconds, len(subjects))


def main(argv: Sequence[str] | None = None) -> int:
    """Score the cached subjects and print provisional tables."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--max-subjects",
        type=int,
        help="Use only the first N cached subjects, to get an answer sooner.",
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Also compute residual_vessels (slow: reruns every detector).",
    )
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    subjects, skipped = cached_subjects(config, arguments.max_subjects)
    if skipped:
        say(f"skipped (incomplete or unreadable cache): {', '.join(skipped)}")
    if not subjects:
        print(
            f"No usable cached subjects in {config.output_dir / 'cache'}. Run the "
            "experiment first; each subject is cached as soon as it is prepared."
        )
        return 1
    say(f"{len(subjects)} cached subjects: {', '.join(name for name, _ in subjects)}")
    say(f"strategies: {list(config.strategies)}; iterations: {list(config.iterations)}")

    rows = analyze(config, subjects, arguments.residual)
    recommended = {
        strategy: recommend(strategy_rows, config.max_lesion_loss, config.auc_tolerance)
        for strategy, strategy_rows in rows.items()
    }
    write_results(config.output_dir, rows, recommended, PARTIAL_RESULTS)

    lesions = int(next(iter(rows.values()))[0]["lesions"])
    print()
    print(f"PROVISIONAL: {len(subjects)} subjects, {lesions} lesions.")
    if lesions < FEW_LESIONS:
        print(
            f"With fewer than {FEW_LESIONS} lesions the AUC and recall values mostly "
            "reflect individual cases; lesion_loss and vessel_fraction are steadier."
        )
    if not arguments.residual:
        print("residual_vessels was skipped (use --residual to compute it).")
    print()
    print(format_table(rows, config.recall_at))
    print()
    for strategy, count in recommended.items():
        if count is None:
            print(f"{strategy}: no iteration count keeps lesion loss within the limit.")
        else:
            print(f"{strategy}: provisionally recommended iterations: {count}")
    print(f"Wrote {config.output_dir / PARTIAL_RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
