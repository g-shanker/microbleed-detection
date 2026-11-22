import re
from pathlib import Path

import numpy as np
import pytest

from experiments.frst.scoring import SubjectScores
from experiments.inpainting import run
from experiments.inpainting.log import RunLog, silent
from experiments.inpainting.measure import Measurement
from experiments.inpainting.tests.support import write_config, write_raw_dataset


def read_log(tmp_path: Path) -> str:
    return (tmp_path / "results" / "run.log").read_text(encoding="utf-8")


def test_silent_discards_messages() -> None:
    assert silent("anything") is None


def test_lines_are_timestamped_printed_and_appended(tmp_path: Path, capsys) -> None:
    log = RunLog(tmp_path / "out" / "run.log")

    log("hello")

    line = (tmp_path / "out" / "run.log").read_text().splitlines()[-1]
    assert re.fullmatch(r"\[\d\d:\d\d:\d\d\] hello", line)
    assert line in capsys.readouterr().out


def test_raw_text_has_no_timestamp(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "run.log")

    log.raw("a table row")

    assert (tmp_path / "run.log").read_text().splitlines()[-1] == "a table row"


def test_a_new_run_appends_after_a_header(tmp_path: Path) -> None:
    RunLog(tmp_path / "run.log")("first")
    RunLog(tmp_path / "run.log")("second")

    lines = (tmp_path / "run.log").read_text().splitlines()
    assert sum(line.startswith("=== run started") for line in lines) == 2
    assert lines[1].endswith("first") and lines[-1].endswith("second")


def test_the_run_log_describes_the_run_and_every_step(
    tmp_path: Path, fake_pipeline
) -> None:
    dataset = write_raw_dataset(tmp_path, ["a"])

    run.main(["--config", str(write_config(tmp_path, dataset))])

    text = read_log(tmp_path)
    assert "strategies: ['axial', 'frangi3d']; iterations: [0, 2]" in text
    assert "subjects: 1; masks saved: True" in text
    assert "[1/1] a" in text
    assert "preprocessing (not cached; runs FSL)" in text
    assert "detected axial in" in text and "detected frangi3d in" in text
    assert "cache written" in text
    assert "axial n=0:" in text and "frangi3d n=2:" in text
    assert "masks saved to" in text
    assert "subject done in" in text and "left" in text


def test_every_measurement_line_reports_what_was_erased(
    tmp_path: Path, fake_pipeline
) -> None:
    dataset = write_raw_dataset(tmp_path, ["a"])

    run.main(["--config", str(write_config(tmp_path, dataset))])

    text = read_log(tmp_path)
    assert "axial n=0: lesion voxels erased 0/1, lesions hit 0/1" in text
    assert "axial n=2: lesion voxels erased 1/1, lesions hit 1/1" in text


def test_the_table_and_recommendations_are_in_the_log(
    tmp_path: Path, fake_pipeline
) -> None:
    dataset = write_raw_dataset(tmp_path, ["a"])

    run.main(["--config", str(write_config(tmp_path, dataset))])

    text = read_log(tmp_path)
    assert "Strategy: axial" in text and "Strategy: frangi3d" in text
    assert "axial: recommended in-plane dilation iterations" in text
    assert "wrote" in text and "results.csv" in text
    assert f"log: {tmp_path / 'results' / 'run.log'}" in text


def test_a_rerun_reports_the_cache_and_appends(tmp_path: Path, fake_pipeline) -> None:
    config = write_config(tmp_path, write_raw_dataset(tmp_path, ["a"]))

    run.main(["--config", str(config)])
    run.main(["--config", str(config)])

    text = read_log(tmp_path)
    assert text.count("=== run started") == 2
    assert "preprocessing: cached" in text
    assert "detection axial: cached" in text


def test_a_new_strategy_is_the_only_detection_after_a_cached_run(
    tmp_path: Path, fake_pipeline
) -> None:
    dataset = write_raw_dataset(tmp_path, ["a"])
    run.main(["--config", str(write_config(tmp_path, dataset, strategies=["axial"]))])

    run.main(["--config", str(write_config(tmp_path, dataset))])

    second_run = read_log(tmp_path).split("=== run started")[-1]
    assert "detection axial: cached" in second_run
    assert "detecting vessels: frangi3d" in second_run
    assert "detecting vessels: axial" not in second_run


def test_a_failure_is_logged_with_its_traceback_and_raised(
    tmp_path: Path, fake_pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args, **kwargs):
        raise RuntimeError("detector exploded")

    monkeypatch.setattr(run, "measure", broken)
    dataset = write_raw_dataset(tmp_path, ["a"])

    with pytest.raises(RuntimeError, match="detector exploded"):
        run.main(["--config", str(write_config(tmp_path, dataset))])

    text = read_log(tmp_path)
    assert "FAILED; the run stopped with this error:" in text
    assert "RuntimeError: detector exploded" in text
    assert "Traceback" in text


def test_a_dry_run_creates_no_log(tmp_path: Path, fake_pipeline) -> None:
    dataset = write_raw_dataset(tmp_path, ["a"])

    run.main(["--config", str(write_config(tmp_path, dataset)), "--dry-run"])

    assert not (tmp_path / "results").exists()


def test_the_progress_estimate_scales_with_the_subjects_left(
    tmp_path: Path, fake_pipeline
) -> None:
    dataset = write_raw_dataset(tmp_path, ["a", "b"])

    run.main(["--config", str(write_config(tmp_path, dataset))])

    done = [
        line for line in read_log(tmp_path).splitlines() if "subject done in" in line
    ]
    assert len(done) == 2
    assert done[-1].endswith("about 0s left")


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(5, "5s"), (65, "1m05s"), (3600, "1h00m"), (7380, "2h03m")],
)
def test_durations_are_formatted_for_people(seconds: float, expected: str) -> None:
    assert run.format_duration(seconds) == expected


def test_a_measurement_is_summarized_for_the_log() -> None:
    measurement = Measurement(
        lesion_voxels=10,
        lesion_voxels_removed=3,
        lesions=4,
        lesions_hit=2,
        vessel_fraction=0.0123,
        residual_vessels=0.5,
        scores=SubjectScores(np.array([]), np.array([])),
    )

    assert run.describe(measurement) == (
        "lesion voxels erased 3/10, lesions hit 2/4, 1.23% of brain inpainted"
    )
