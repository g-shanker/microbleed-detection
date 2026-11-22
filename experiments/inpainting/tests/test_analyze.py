import csv
from pathlib import Path

import numpy as np
import pytest

from experiments.inpainting import analyze, detectors, run
from experiments.inpainting.config import load_config
from experiments.inpainting.tests.support import write_config, write_raw_dataset
from microbleednet.core.engines import processor


@pytest.fixture
def finished_run(tmp_path: Path, fake_pipeline) -> Path:
    """A completed run over two subjects, leaving its cache behind."""
    dataset = write_raw_dataset(tmp_path, ["a", "b"])
    config = write_config(tmp_path, dataset)
    run.main(["--config", str(config)])
    (tmp_path / "results" / "results.csv").unlink()
    return config


def read_partial(config: Path) -> list[dict[str, str]]:
    path = load_config(config).output_dir / "partial_results.csv"
    with path.open() as file:
        return list(csv.DictReader(file))


def forbid_recomputing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        processor, "preprocess", lambda _: pytest.fail("preprocessing ran")
    )
    for name in detectors.DETECTORS:
        monkeypatch.setitem(
            detectors.DETECTORS, name, lambda _: pytest.fail("detector ran")
        )


def test_it_scores_the_cached_subjects_without_preprocessing_or_detecting(
    finished_run: Path, monkeypatch
) -> None:
    forbid_recomputing(monkeypatch)

    assert analyze.main(["--config", str(finished_run)]) == 0

    rows = read_partial(finished_run)
    assert [(r["strategy"], r["iterations"]) for r in rows] == [
        ("axial", "0.0"),
        ("axial", "2.0"),
        ("frangi3d", "0.0"),
        ("frangi3d", "2.0"),
    ]
    assert all(row["subjects"] == "2.0" for row in rows)


def test_it_reproduces_the_full_runs_numbers(
    tmp_path: Path, finished_run: Path, monkeypatch
) -> None:
    run.main(["--config", str(finished_run)])
    with (tmp_path / "results" / "results.csv").open() as file:
        full = list(csv.DictReader(file))
    forbid_recomputing(monkeypatch)

    analyze.main(["--config", str(finished_run)])

    for partial, complete in zip(read_partial(finished_run), full, strict=True):
        for column in ("lesion_loss", "lesions_hit", "vessel_fraction", "mean_auc"):
            assert partial[column] == complete[column]


def test_residual_vessels_are_skipped_unless_requested(
    finished_run: Path, capsys
) -> None:
    analyze.main(["--config", str(finished_run)])

    assert {row["residual_vessels"] for row in read_partial(finished_run)} == {"nan"}
    assert "residual_vessels was skipped" in capsys.readouterr().out


def test_residual_vessels_are_computed_on_request(finished_run: Path, capsys) -> None:
    analyze.main(["--config", str(finished_run), "--residual"])

    assert "nan" not in {r["residual_vessels"] for r in read_partial(finished_run)}
    assert "skipped" not in capsys.readouterr().out


def test_the_number_of_subjects_can_be_limited(finished_run: Path) -> None:
    analyze.main(["--config", str(finished_run), "--max-subjects", "1"])

    assert {row["subjects"] for row in read_partial(finished_run)} == {"1.0"}


def test_the_output_is_labelled_provisional_with_the_lesion_count(
    finished_run: Path, capsys
) -> None:
    analyze.main(["--config", str(finished_run)])

    output = capsys.readouterr().out
    assert "PROVISIONAL: 2 subjects, 2 lesions." in output
    assert "fewer than 10 lesions" in output
    assert "Strategy: axial" in output and "Strategy: frangi3d" in output
    assert "axial: provisionally recommended iterations" in output


def test_it_reports_progress_with_a_time_estimate(finished_run: Path, capsys) -> None:
    analyze.main(["--config", str(finished_run)])

    output = capsys.readouterr().out
    assert "2 cached subjects: a, b" in output
    assert "[2/2] b" in output
    assert "about 0s left" in output


def test_it_writes_only_the_partial_table(finished_run: Path) -> None:
    analyze.main(["--config", str(finished_run)])

    results = load_config(finished_run).output_dir
    assert (results / "partial_results.csv").is_file()
    assert not (results / "results.csv").exists()


def test_it_does_not_touch_the_runs_log(finished_run: Path) -> None:
    log = load_config(finished_run).output_dir / "run.log"
    before = log.read_text()

    analyze.main(["--config", str(finished_run)])

    assert log.read_text() == before


def test_a_subject_missing_a_strategy_is_skipped(
    tmp_path: Path, fake_pipeline, capsys
) -> None:
    dataset = write_raw_dataset(tmp_path, ["a", "b"])
    run.main(["--config", str(write_config(tmp_path, dataset, strategies=["axial"]))])
    config = write_config(tmp_path, dataset)

    code = analyze.main(["--config", str(config)])

    assert code == 1
    assert "skipped (incomplete or unreadable cache): a, b" in capsys.readouterr().out


def test_an_unreadable_cache_file_is_skipped(finished_run: Path, capsys) -> None:
    cache = load_config(finished_run).output_dir / "cache" / "fast"
    (cache / "b.npz").write_bytes(b"not a real archive")

    analyze.main(["--config", str(finished_run)])

    output = capsys.readouterr().out
    assert "skipped (incomplete or unreadable cache): b" in output
    assert {row["subjects"] for row in read_partial(finished_run)} == {"1.0"}


def test_files_being_written_are_ignored(finished_run: Path) -> None:
    cache = load_config(finished_run).output_dir / "cache" / "fast"
    (cache / "c.partial.npz").write_bytes(b"half written")

    analyze.main(["--config", str(finished_run)])

    assert {row["subjects"] for row in read_partial(finished_run)} == {"2.0"}


def test_it_explains_when_nothing_has_been_cached(tmp_path: Path, capsys) -> None:
    dataset = write_raw_dataset(tmp_path, ["a"])

    code = analyze.main(["--config", str(write_config(tmp_path, dataset))])

    assert code == 1
    assert "No usable cached subjects" in capsys.readouterr().out


def test_many_lesions_need_no_warning(
    finished_run: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analyze, "FEW_LESIONS", 1)

    analyze.main(["--config", str(finished_run)])

    assert "fewer than" not in capsys.readouterr().out


def test_no_recommendation_is_reported_when_every_count_loses_too_much(
    tmp_path: Path, fake_pipeline, capsys
) -> None:
    dataset = write_raw_dataset(tmp_path, ["a"])
    config = write_config(
        tmp_path, dataset, iterations=[2], strategies=["axial"], max_lesion_loss=0.0
    )
    run.main(["--config", str(config)])

    analyze.main(["--config", str(config)])

    assert "axial: no iteration count keeps lesion loss" in capsys.readouterr().out


def test_cache_entries_load_in_name_order(finished_run: Path) -> None:
    config = load_config(finished_run)

    subjects, skipped = analyze.cached_subjects(config, None)

    assert [name for name, _ in subjects] == ["a", "b"] and not skipped
    assert all(np.asarray(p.vessels["axial"]).any() for _, p in subjects)
